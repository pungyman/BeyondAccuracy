"""vLLM inference backend for the intervention arms — continuous batching without losing the hook seam.

Why this exists: the generative MGSM path is dominated by *straggler* batches — HF `generate`
runs a batch until its slowest member hits EOS or the 512-token cap, so one rambling Bengali
sample idles 127 other rows. vLLM's continuous batching refills those slots. The blocker was
believed to be that interventions are forward-hook-based; that belief was wrong. vLLM's decoder
layers are ordinary `nn.Module`s and accept `register_forward_hook`, which is exactly how Zhao
et al. steer (`mlrs/vllms/ForSteering_vllm.py`, class `ForSteeringVLLM`).

TWO CORRECTNESS TRAPS this module handles, both invisible until you diff against HF:

1. **vLLM defers the residual add.** An HF decoder layer returns the true residual stream. A vLLM
   layer returns `(hidden_states, residual)` where the *stream* is `hidden_states + residual` —
   the add is fused into the next layer's RMSNorm for speed. Editing `output[0]` alone (what
   Zhao's `def_hook_steer_fn` does) therefore edits only the branch output, NOT the residual
   stream, and is **not** equivalent to the HF-path ablation this repo measures. We reconstruct
   `s = hidden_states + residual`, project *that*, and subtract the delta from `hidden_states`
   so the next layer sees exactly `s - λ·P·s`. Models that return a bare tensor (no deferred
   residual) are handled by the plain path.

2. **Layer identity must be bound, not counted.** Zhao infer the current layer from a global
   call counter mod `layer_num`; that desynchronizes the moment two forward passes interleave or
   a hook raises. We bind the layer index in the closure, like `intervene.build_layer_hooks`.

Scope: `prompt_only` (the reported setting, and Zhao's `steering_level="prompt"`) means edit the
prompt tokens only. This used to be decided per *forward* — "does the `positions` tensor contain a
zero?" — which is sound only when a prompt is never split and never batched with a decode step.
vLLM 0.8.5's V1 engine breaks both halves of that (`open.md` Q3): `_set_default_args_v1` sets
`enable_chunked_prefill = True` unconditionally, overriding the `False` we pass with no warning, so
a continuation chunk carries no zero and would be skipped as if it were decode. The decision is
therefore made **per token**, from the engine's own batch metadata — see `prompt_token_mask`. The
old heuristic survives only as the last-resort fallback when no metadata is reachable at all.

The engine's second V1 default, prefix caching, cannot be masked around: a cached prefix means the
edited layers never re-run for it, so the edit is not clipped but absent for most of most prompts.
`VllmRunner` passes `enable_prefix_caching=False` and, for a `prompt_only` spec, *verifies* that the
built config agrees before registering a single hook.

Nothing here is trusted until `scripts/validate_vllm_vs_hf.py` shows per-item agreement with the
HF path — kernel and batching differences are expected, silent semantic drift is not.
"""
from __future__ import annotations

import contextlib
import os

import pandas as pd
import torch

from corpus import build_chat_inputs
from intervene import AffineSpec, InterventionSpec
from progress import Progress
from scoring import (detlang, extract_mgsm_answer, is_correct, script_dom,
                     strip_reasoning)

_INNER_MODEL_PATHS = (
    # vLLM V1 (>=0.8): the engine core owns the model runner
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model",
    "llm_engine.engine_core.model_executor.driver_worker.model_runner.model",
    # vLLM V0 (Zhao's path)
    "llm_engine.model_executor.driver_worker.model_runner.model",
)


def _dig(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_inner_model(llm):
    """The real `nn.Module` behind an `LLM`, across vLLM engine versions."""
    errors = []
    for path in _INNER_MODEL_PATHS:
        try:
            return _dig(llm, path)
        except AttributeError as e:  # noqa: PERF203 - probing distinct layouts
            errors.append(f"{path}: {e}")
    raise RuntimeError(
        "could not reach vLLM's inner model; tried:\n  " + "\n  ".join(errors)
        + "\nInspect `llm.llm_engine` and add the layout to _INNER_MODEL_PATHS."
    )


def resolve_model_runner(llm):
    """vLLM's model runner (the owner of `input_batch`), or None if the layout is unknown.

    Every entry in `_INNER_MODEL_PATHS` is `<runner>.model`, so the runner is that path minus its
    last hop. V1 needs it for the exact prompt-token mask: `input_batch.num_prompt_tokens` is the
    only place the *prompt length* of each in-flight request is written down, and without it a
    chunked prefill cannot be told from a decode step. Returns None rather than raising — a clean
    (spec-free) run never needs this, and `interventions()` raises with a better message.
    """
    for path in _INNER_MODEL_PATHS:
        try:
            return _dig(llm, path.rsplit(".", 1)[0])
        except AttributeError:
            continue
    return None


# `model.layers` / `layers` cover every text-only arch in the ladder. The `language_model` entries
# are vLLM's multimodal layout: Gemma 3 12B/27B register as `Gemma3ForConditionalGeneration`, so
# the decoder blocks hang off the text tower rather than the top-level module.
_LAYER_PATHS = (
    "model.layers",
    "layers",
    "language_model.model.layers",
    "model.language_model.layers",
    "transformer.h",
    "model.decoder.layers",
)


def resolve_layers(inner):
    """The decoder-layer ModuleList (`model.layers` on text-only archs; deeper on multimodal)."""
    for path in _LAYER_PATHS:
        try:
            layers = _dig(inner, path)
        except AttributeError:
            continue
        if hasattr(layers, "__len__") and len(layers):
            return layers
    raise RuntimeError(f"no decoder-layer list found on {type(inner).__name__}")


def _is_prefill(positions) -> bool:
    """True when this forward looks like a prompt (prefill) pass — the FALLBACK rule only.

    Decode steps continue sequences, so every position is >=1; an unsplit prefill batch always
    contains position 0. That reasoning holds only with chunked prefill off, which V1 does not
    allow, so this is used only when `prompt_token_mask` cannot reach the engine's metadata.
    """
    if positions is None or not torch.is_tensor(positions) or positions.numel() == 0:
        return True                                    # unknown layout: treat as prompt (edit)
    return bool((positions == 0).any())


def _unwrap_attn_metadata(md):
    """One metadata object, whatever the version hands us (>=0.9 keys it per layer name)."""
    if isinstance(md, dict):
        return next(iter(md.values())) if md else None
    return md


def prompt_token_mask(positions, attn_metadata, num_prompt_tokens=None):
    """Per-token bool mask over the flattened batch: True where the token belongs to a PROMPT.

    This is what `prompt_only` actually means once continuous batching is in play — a single
    forward can carry one request's prefill chunk next to another's decode step, so the question
    is per token, never per forward. Returns None when the metadata cannot answer it; the caller
    then falls back to `_is_prefill`.

    Two engine layouts, both exact:

    * **V0 (vLLM 0.7.x)** orders the batch prefill-tokens-first and publishes the split point as
      `num_prefill_tokens`, so the mask is a prefix. Trailing CUDA-graph padding falls outside it
      and is correctly False.
    * **V1 (vLLM >=0.8)** flattens requests in `input_batch` order with `query_start_loc` as the
      boundaries, and `positions` already holds each token's ABSOLUTE index in its sequence
      (`gpu_model_runner._prepare_inputs` computes it as `num_computed_tokens[req] + arange`).
      A token is therefore a prompt token exactly when its position is below its own request's
      prompt length. That is what makes this robust where a query-length heuristic is not: a
      chunked prefill's final chunk can be a single token, which is indistinguishable from a
      decode step by shape alone but differs by one in position.

    `num_prompt_tokens` is `input_batch.num_prompt_tokens` (indexed by request slot). Anything
    unexpected — no metadata, no prompt lengths, a token count that disagrees with the batch —
    returns None rather than a guess, because a wrong mask silently mis-edits.
    """
    md = _unwrap_attn_metadata(attn_metadata)
    if md is None or positions is None or not torch.is_tensor(positions):
        return None
    n_tokens = int(positions.numel())
    if n_tokens == 0:
        return None

    n_prefill = getattr(md, "num_prefill_tokens", None)
    if n_prefill is not None:                          # V0: an exact prefix of the batch
        idx = torch.arange(n_tokens, device=positions.device)
        return idx < int(n_prefill)

    qsl = getattr(md, "query_start_loc", None)         # V1
    if qsl is None or num_prompt_tokens is None or not torch.is_tensor(qsl) or qsl.numel() < 2:
        return None
    qsl = qsl.to(device=positions.device, dtype=torch.long)
    query_lens = qsl[1:] - qsl[:-1]
    n_reqs = int(query_lens.numel())
    if n_reqs > len(num_prompt_tokens):
        return None                                    # metadata wider than the batch: don't guess
    n_real = int(query_lens.sum().item())
    if n_real <= 0 or n_real > n_tokens:
        return None
    plens = torch.as_tensor(list(num_prompt_tokens[:n_reqs]),
                            device=positions.device, dtype=positions.dtype)
    per_token_prompt_len = torch.repeat_interleave(plens, query_lens)
    mask = torch.zeros(n_tokens, dtype=torch.bool, device=positions.device)
    mask[:n_real] = positions[:n_real] < per_token_prompt_len
    return mask                                        # tail (CUDA-graph padding) stays False


def _apply_edit(stream, fn, mask):
    """`fn` over the whole tensor, kept only where `mask` says the token is a prompt token."""
    edited = fn(stream)
    if mask is None:
        return edited
    if mask.numel() != stream.shape[0]:                # unexpected layout: all-or-nothing
        return edited if bool(mask.any()) else stream
    return torch.where(mask.view(-1, *([1] * (stream.dim() - 1))), edited, stream)


def _layer_hook(fn, scope: str, mask_fn=None):
    """Adapt an `intervene` edit `fn(h)->h` into a vLLM forward hook.

    Handles the deferred-residual layout: when a layer returns `(hidden_states, residual)`, the
    residual stream is their sum, so we project the SUM and subtract that delta from
    `hidden_states` — leaving `hidden_states + residual` equal to the intended edited stream.

    `mask_fn(positions) -> mask | None` scopes a `prompt_only` edit to the prompt tokens of a
    mixed batch. None from it means "no metadata" and drops back to the whole-forward rule.
    """
    def hook(module, inputs, output):
        mask = None
        if scope == "prompt_only":
            positions = inputs[0] if inputs else None
            mask = mask_fn(positions) if mask_fn is not None else None
            if mask is None:
                if not _is_prefill(positions):
                    return output                      # decode step: prompt-only edits nothing
            elif not bool(mask.any()):
                return output                          # no prompt token in this batch

        if isinstance(output, tuple):
            hidden = output[0]
            residual = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else None
            if residual is not None and residual.shape == hidden.shape:
                stream = hidden + residual             # the true residual stream
                edited = _apply_edit(stream, fn, mask)
                return (hidden + (edited - stream),) + tuple(output[1:])
            return (_apply_edit(hidden, fn, mask),) + tuple(output[1:])
        return _apply_edit(output, fn, mask)

    return hook


class VllmRunner:
    """A warm vLLM engine whose decoder layers accept this repo's intervention specs."""

    def __init__(self, model_name: str, *, dtype: str = "bfloat16",
                 max_model_len: int = 8192, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85, enforce_eager: bool = False,
                 seed: int = 0):
        # max_model_len default: 8192, NOT 4096. The 8-shot MGSM prompt is ~2k tokens only in
        # English; on byte-fallback BPEs the Telugu prompt reaches 4.9k (Qwen2.5), 5.5k (Mistral),
        # 5.7k (Llama-3.1). vLLM 0.7 admits over-long TEXT prompts without error and finishes them
        # immediately (finish_reason="length", 0 tokens), which silently zeroed every te row of the
        # first E7 campaign. 8192 covers the longest prompt + the 512 cap and is still within every
        # roster model's context (gemma-2's hard limit is exactly 8192). KV-pool size is set by
        # gpu_memory_utilization, not by this cap — raising it costs no VRAM.
        # The hook seam needs the model IN THIS PROCESS. vLLM 0.8.5's V1 engine defaults to
        # running EngineCore -- and therefore the model -- in a child process (`SyncMPClient`), so
        # no attribute walk from `LLM` can reach it and `resolve_inner_model` fails outright
        # (job 301937, 2026-09-03). Disabling V1 multiprocessing puts EngineCore back in-process
        # (`InprocClient`), which is what `_INNER_MODEL_PATHS[0]` was written for, and removes the
        # forked-CUDA hazard (job 301929) at the same time: there is no fork left to poison.
        # 0.7.3 ran V0 in-process and never needed this. `setdefault`, so an explicit override
        # still wins -- and then fails loudly at `resolve_inner_model` rather than silently.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM                            # heavy; import only when used
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.enforce_eager = bool(enforce_eager)
        self.llm = LLM(
            model=model_name, dtype=dtype, max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            # V0 honours this and the batch stays prefill-then-decode; V1 overrides it to True
            # regardless (`_set_default_args_v1`), which is why the hook masks per token.
            enable_chunked_prefill=False,
            # NOT a default we can leave alone. V1 turns prefix caching ON unless told otherwise,
            # and the 8-shot MGSM prompts share a long identical prefix within a language, so from
            # the second item on the shared prefix would be served from cached KV — the edited
            # layers never re-run for it and the intervention is absent from most of most prompts.
            # No mask can recover that, so it has to be off at the engine. V1 respects an explicit
            # False (the override is guarded on `is None`); `_assert_prompt_only_seam` checks that
            # it actually took rather than trusting it.
            enable_prefix_caching=False,
            enforce_eager=enforce_eager,                # prefill runs eager regardless; graphs cover decode
            seed=seed,
        )
        self.inner = resolve_inner_model(self.llm)
        self.model_runner = resolve_model_runner(self.llm)
        self.layers = resolve_layers(self.inner)
        self.n_layers = len(self.layers)
        self.engine_flags = self._read_engine_flags()

    def _read_engine_flags(self) -> dict:
        """What the engine ACTUALLY does, read back from the built config, not from what we asked.

        Both flags that matter here are silently rewritten by V1, so asking the config is the only
        honest way to know; `getattr` chains because the config moved between 0.7 and 0.8.
        """
        engine = getattr(self.llm, "llm_engine", None)
        cfg = getattr(engine, "vllm_config", None)
        sched = getattr(cfg, "scheduler_config", None) or getattr(engine, "scheduler_config", None)
        cache = getattr(cfg, "cache_config", None) or getattr(engine, "cache_config", None)
        return {
            "v1": hasattr(engine, "engine_core"),
            "chunked_prefill": bool(getattr(sched, "chunked_prefill_enabled",
                                            getattr(sched, "enable_chunked_prefill", False))),
            "prefix_caching": bool(getattr(cache, "enable_prefix_caching", False)),
            "exact_mask": getattr(getattr(self.model_runner, "input_batch", None),
                                  "num_prompt_tokens", None) is not None,
        }

    def _prompt_mask(self, positions):
        """The per-token prompt mask for the forward now running, or None if it can't be built."""
        try:
            from vllm.forward_context import get_forward_context
            md = get_forward_context().attn_metadata
        except Exception:                              # no context (V0 profiling run, or a stub)
            return None
        ib = getattr(self.model_runner, "input_batch", None)
        return prompt_token_mask(positions, md, getattr(ib, "num_prompt_tokens", None))

    def _assert_prompt_only_seam(self) -> None:
        """Refuse a `prompt_only` edit the engine would silently drop (`open.md` Q3).

        Registering hooks and discovering afterwards that fidelity did not move is the failure mode
        this exists to prevent: a dropped edit looks exactly like a weak one, and the arm reads as a
        null result rather than as a broken run.
        """
        f = self.engine_flags
        if f["prefix_caching"]:
            raise RuntimeError(
                "vLLM has prefix caching ENABLED, which is incompatible with a prompt_only "
                "intervention: the shared 8-shot prefix would be served from cached KV and the "
                "edited layers would never run for it, so the edit would be absent from most of "
                "most prompts rather than merely clipped. VllmRunner passes "
                "enable_prefix_caching=False; this engine build ignored it.")
        if f["chunked_prefill"] and not f["exact_mask"]:
            raise RuntimeError(
                "vLLM has chunked prefill ENABLED (V1 forces it) but the model runner's "
                "input_batch.num_prompt_tokens is not reachable, so a prompt chunk cannot be told "
                "from a decode step and a prompt_only edit would be dropped on every chunk that "
                "does not start at position 0. Add this engine's layout to _INNER_MODEL_PATHS, or "
                "run the arm on the V0 engine (vLLM 0.7.3).")

    @contextlib.contextmanager
    def interventions(self, spec):
        """Register `spec`'s per-layer edits for the duration of the block (no-op if `spec` None)."""
        if spec is None:
            yield
            return
        from intervene import build_affine_hooks, build_layer_hooks
        if spec.scope == "prompt_only":
            self._assert_prompt_only_seam()
        elif not self.enforce_eager:
            # Decode steps replay captured CUDA graphs, which never call Python forward hooks, so an
            # every-step edit would fire on the prompt only -- all_steps would silently equal
            # prompt_only. E11's p3a hit exactly this: 4,000 answers byte-identical to the
            # prompt-only run (2026-09-22). Refuse rather than mislabel.
            raise ValueError(
                f"scope={spec.scope!r} needs an eager vLLM engine: decode steps run as CUDA graphs "
                f"that skip Python hooks, so the edit would apply to the prompt only. Build the "
                f"engine with enforce_eager=True (set E5_VLLM_ENFORCE_EAGER=1).")
        build = build_affine_hooks if isinstance(spec, AffineSpec) else build_layer_hooks
        fns = build(spec, mask_holder=None)             # {layer: fn(h)->h}; scope handled per-hook
        handles = []
        try:
            for idx, fn in fns.items():
                handles.append(self.layers[int(idx)].register_forward_hook(
                    _layer_hook(fn, spec.scope, mask_fn=self._prompt_mask)))
            yield
        finally:
            for h in handles:
                h.remove()

    def generate(self, prompts, *, max_new_tokens: int, spec=None, desc: str = "",
                 temperature: float = 0.0, top_p: float = 1.0, top_k: int = -1,
                 repetition_penalty: float | None = None, seed: int | None = None):
        """Greedy (or sampled) continuations for `prompts`, with `spec` applied via hooks."""
        from vllm import SamplingParams
        sp_kw = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                     max_tokens=max_new_tokens, n=1, seed=seed)
        if repetition_penalty is not None:
            sp_kw["repetition_penalty"] = repetition_penalty
        sp = SamplingParams(**sp_kw)
        prog = Progress(len(prompts), desc or "vllm", min_interval=60.0)
        with self.interventions(spec):
            outs = self.llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        prog.update(len(prompts), info="generated")
        prog.close()
        return outs


def check_context_window(tok, prompts, *, max_new_tokens: int, max_model_len: int) -> int:
    """Fail fast when any templated prompt cannot fit `max_new_tokens` inside `max_model_len`.

    vLLM (0.7, text models) does NOT reject over-long prompts — it admits them and finishes the
    request immediately with finish_reason="length" and zero generated tokens, so an undersized
    window shows up as silently empty outputs, not as an error. Returns the longest prompt length.
    """
    longest = max(len(tok.encode(p)) for p in prompts)
    if longest + max_new_tokens > max_model_len:
        raise ValueError(
            f"longest templated prompt is {longest} tokens; +{max_new_tokens} new tokens "
            f"exceeds the engine's max_model_len={max_model_len}. vLLM would emit EMPTY "
            f"generations for those items (finish_reason='length', 0 tokens) instead of "
            f"erroring. Raise E5_VLLM_MAX_MODEL_LEN (or VllmRunner(max_model_len=...)) to at "
            f"least {longest + max_new_tokens}.")
    return longest


def run_eval_vllm(runner: VllmRunner, tok, items, *, spec=None, cond: str | None = None,
                  max_new_tokens: int | None = None, model_name: str | None = None,
                  enable_thinking: bool = False, gen=None) -> pd.DataFrame:
    """`evaluate.run_eval`'s generative path on vLLM — same row schema, same scoring functions.

    Deliberately reuses `build_chat_inputs` and the `scoring` helpers so templating and answer
    extraction cannot drift from the HF path; only the generation engine differs.
    """
    tag = (model_name or runner.model_name).split("/")[-1]
    cond = cond or ("clean" if spec is None else "intervention")
    max_new_tokens = max_new_tokens or int(os.environ.get("E5_MAX_NEW_TOKENS", "512"))
    prompts = build_chat_inputs(tok, [it.prompt for it in items],
                                enable_thinking=enable_thinking)
    check_context_window(tok, prompts, max_new_tokens=max_new_tokens,
                         max_model_len=runner.max_model_len)
    kw = {}
    if gen is not None and getattr(gen, "do_sample", False):
        kw = {"temperature": gen.temperature, "top_p": gen.top_p, "seed": gen.seed}
        if gen.top_k is not None:
            kw["top_k"] = gen.top_k
        if gen.repetition_penalty is not None:
            kw["repetition_penalty"] = gen.repetition_penalty
    outs = runner.generate(prompts, max_new_tokens=max_new_tokens, spec=spec,
                           desc=f"vllm[{cond} n={len(items)}]", **kw)

    keep_trace = os.environ.get("E5_KEEP_TRACE") not in (None, "", "0")
    rows = []
    for it, out in zip(items, outs):
        o = out.outputs[0]
        full = o.text
        n_new = len(o.token_ids)
        truncated = bool(o.finish_reason == "length")
        answer = strip_reasoning(full, expect_trace=enable_thinking)
        close = full.rfind("</think>")
        if close != -1:
            reasoning, has_trace = full[:close].replace("<think>", ""), True
        elif enable_thinking or "<think>" in full:
            reasoning, has_trace = full.replace("<think>", ""), True
        else:
            reasoning, has_trace = "", False
        pred = extract_mgsm_answer(answer, it.lang)
        rows.append({
            "task": it.task, "lang": it.lang, "id": it.meta.get("id"),
            "pred": pred, "gold": it.gold,
            "correct": bool(is_correct(pred, it.gold)),
            "output_script": script_dom(answer),
            "output_lang": detlang(answer),
            "reasoning_lang": detlang(reasoning) if has_trace else "",
            "n_new_tokens": n_new, "truncated": truncated,
            "max_new_tokens": int(max_new_tokens),
            "output": (full if keep_trace else answer),
        })
    df = pd.DataFrame(rows)
    df.insert(0, "model", tag)
    df["cond"] = cond
    df["engine"] = "vllm"        # open.md Q1: engine provenance, stamped at the source
    return df
