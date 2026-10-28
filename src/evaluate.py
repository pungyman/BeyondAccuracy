"""Reusable E5 evaluation harness — the clean (no-subspace) baseline path (P1).

`run_eval` turns a list of `BenchItem`s into a tidy per-item `DataFrame`, dispatching on task:

- **Generative (MGSM):** length-bucketed greedy `model.generate` (the batching approach from
  `extract.extract_last_token_per_layer`), then `is_correct(extract_mgsm_answer(out, lang), gold)`.
  Also logs `script_dom` / `detlang` for the E5 §7 output-language fidelity axis.
- **Likelihood (M-MMLU, XWinograd):** `score_continuations` runs a teacher-forced forward per
  option, sums the continuation token log-probs, and argmaxes — no sampling.

The `hooks` argument is the **P2 intervention seam**: `None` = clean baseline; P2 passes
`build_layer_hooks(spec)` and the same code path applies it via `intervene.interventions(model,
hooks)`. For the likelihood path, position scoping ("prompt-only" edits only context tokens) is
carried by a shared `intervene.MaskHolder` the harness sets per teacher-forced forward from the
context length; pass it alongside `hooks` (both are built from the same `InterventionSpec`).
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
import torch

from config import get_device, results_path
from corpus import build_chat_inputs
from extract import _mps_empty_cache, _pack_batches
from geometry import bootstrap_ci, bootstrap_ci_cluster
from intervene import interventions
from models import model_spec
from scoring import (
    detlang,
    extract_mgsm_answer,
    is_correct,
    is_correct_mc,
    score_cloze,
    script_dom,
    strip_reasoning,
)

_GENERATIVE = {"mgsm"}
_LIKELIHOOD = {"mmlu", "xwinograd"}


def _hook_ctx(model, hooks):
    """`intervene.interventions` when hooks are given, else a no-op (clean baseline)."""
    return interventions(model, hooks) if hooks else contextlib.nullcontext()


# ---- vLLM engine routing (opt-in via E5_ENGINE=vllm) --------------------------------------------
# A vLLM engine costs ~1-2 min to construct, but a λ-search makes hundreds of run_eval calls, so the
# engine MUST be built once per process and reused. This module-level lazy singleton (keyed by the
# full model name) is that reuse: the FIRST run_eval that needs it builds the engine, every later
# call — every search trial, every clean/ladder eval — shares the warm one. Default E5_ENGINE=hf
# leaves the entire path byte-identical; nothing below imports vllm unless the flag is on.
_VLLM_RUNNERS: dict = {}


def _use_vllm() -> bool:
    """True when the generative MGSM path should run on the vLLM engine (E5_ENGINE=vllm)."""
    return os.environ.get("E5_ENGINE", "hf").strip().lower() == "vllm"


def _get_vllm_runner(model_name: str):
    """Warm vLLM engine for `model_name`, built once per process and cached (see block comment).

    KV-cache window (`max_model_len`) and VRAM budget are env-tunable so a driver/SLURM job can size
    them per card without code edits: E5_VLLM_MAX_MODEL_LEN (default from VllmRunner = 8192 — the
    8-shot MGSM prompt is ~2k tokens in English but up to 5.7k in Telugu on byte-fallback BPEs;
    4096 silently zeroed every te generation), E5_VLLM_GPU_MEM_UTIL, E5_VLLM_TP (tensor-parallel
    size), E5_VLLM_ENFORCE_EAGER. run_eval_vllm fail-fasts if a prompt cannot fit the window.
    """
    runner = _VLLM_RUNNERS.get(model_name)
    if runner is None:
        from vllm_backend import VllmRunner  # heavy (imports vllm); only when the flag is on
        kw: dict = {}
        if os.environ.get("E5_VLLM_MAX_MODEL_LEN"):
            kw["max_model_len"] = int(os.environ["E5_VLLM_MAX_MODEL_LEN"])
        if os.environ.get("E5_VLLM_GPU_MEM_UTIL"):
            kw["gpu_memory_utilization"] = float(os.environ["E5_VLLM_GPU_MEM_UTIL"])
        if os.environ.get("E5_VLLM_TP"):
            kw["tensor_parallel_size"] = int(os.environ["E5_VLLM_TP"])
        if os.environ.get("E5_VLLM_ENFORCE_EAGER") not in (None, "", "0"):
            kw["enforce_eager"] = True
        runner = VllmRunner(model_name, **kw)
        _VLLM_RUNNERS[model_name] = runner
    return runner


def _model_tag(model, model_name: str | None) -> str:
    """Short model tag for artifact stamping (e.g. 'Qwen2.5-1.5B-Instruct')."""
    name = model_name or getattr(getattr(model, "config", None), "_name_or_path", "") or "model"
    return str(name).split("/")[-1]


@torch.no_grad()
def score_continuations(
    model, tok, context: str, continuations, *,
    hooks=None, device=None, char_norm: bool = False, token_norm: bool = False,
    mask_holder=None,
) -> list[float]:
    """Teacher-forced summed log-prob of each `continuation` given `context`.

    Runs one forward per `context + continuation`, gathers the log-probs of exactly the
    continuation tokens, and sums them (optionally length-normalized). `char_norm` divides by the
    continuation's character length, `token_norm` by its token count — the two lighteval toggles,
    exposed because normalization changes the argmax. No sampling.

    When `mask_holder` (an `intervene.MaskHolder`) is given, its position mask is set from the
    context length before each forward and cleared after, so a `prompt_only` intervention edits
    only the `ctx_len` context tokens — the scored continuation tokens pass through untouched.
    """
    device = device or get_device()
    ctx_ids = tok(context, add_special_tokens=True)["input_ids"]
    ctx_len = len(ctx_ids)
    scores: list[float] = []
    for cont in continuations:
        full_ids = tok(context + cont, add_special_tokens=True)["input_ids"]
        start = max(ctx_len, 1)                             # need a preceding token to predict from
        input_ids = torch.tensor([full_ids], device=device)
        if mask_holder is not None:
            mask_holder.set_from_ctx(len(full_ids), ctx_len)
        try:
            with _hook_ctx(model, hooks):
                logits = model(input_ids).logits[0].float()    # (T, V)
        finally:
            if mask_holder is not None:
                mask_holder.clear()
        logprobs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for t in range(start, len(full_ids)):
            total += float(logprobs[t - 1, full_ids[t]])
        n_tok = max(len(full_ids) - len(ctx_ids), 1)
        if token_norm:
            total /= n_tok
        elif char_norm and len(cont) > 0:
            total /= len(cont)
        scores.append(total)
    return scores


@torch.no_grad()
def _run_generative(
    model, tok, items, *, hooks, batch_size, max_new_tokens, device, max_batch_tokens,
    enable_thinking: bool = False, gen=None, eos_token_id=None,
) -> list[dict]:
    prompts = [it.prompt for it in items]
    templated = build_chat_inputs(tok, prompts, enable_thinking=enable_thinking)
    lengths = [len(ids) for ids in tok(templated)["input_ids"]]
    order = sorted(range(len(templated)), key=lambda j: lengths[j])
    packed = _pack_batches(order, lengths, batch_size, max_batch_tokens)

    keep_trace = os.environ.get("E5_KEEP_TRACE") not in (None, "", "0")
    rows: list[dict | None] = [None] * len(items)
    for bi, sel in enumerate(packed):
        batch = [templated[j] for j in sel]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        # Decoding: greedy by default (do_sample=False, byte-identical to before); when a sampling
        # `gen` is given (reasoning models), seed per batch so the clean and intervention arms —
        # which have identical length-bucketed packing — draw matched noise (paired-delta friendly).
        sampling = {"do_sample": False}
        if gen is not None and gen.do_sample:
            torch.manual_seed(gen.seed + bi)
            sampling = {"do_sample": True, "temperature": gen.temperature, "top_p": gen.top_p}
            if gen.top_k is not None:
                sampling["top_k"] = gen.top_k
        extra = {"eos_token_id": eos_token_id} if eos_token_id is not None else {}
        with _hook_ctx(model, hooks):
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id, **extra, **sampling,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        gen_len = int(new.shape[1])
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for k, j in enumerate(sel):
            it = items[j]
            full = texts[k]
            # differential-truncation instrument (E7 gap #8): a row that filled the whole cap with
            # no trailing pad never emitted EOS -> truncated. Aggressive-ablation arms ramble longer
            # and clip more, biasing paired deltas toward the gentler arm, so this must be auditable.
            n_new = int((new[k] != tok.pad_token_id).sum())
            truncated = bool(gen_len >= max_new_tokens and n_new >= max_new_tokens)
            answer = strip_reasoning(full, expect_trace=enable_thinking)   # truncated trace -> ""
            # reasoning-trace text (for its language): a trace was emitted if </think> is present,
            # OR this is a reasoning-model output — the forced-<think> models (GLM-Z1, R1-distill)
            # put the opening <think> PROMPT-side, so a truncated continuation carries neither tag
            # yet is entirely (truncated) reasoning.
            close = full.rfind("</think>")
            if close != -1:
                reasoning, has_trace = full[:close].replace("<think>", ""), True
            elif enable_thinking or "<think>" in full:
                reasoning, has_trace = full.replace("<think>", ""), True   # truncated trace
            else:
                reasoning, has_trace = "", False
            pred = extract_mgsm_answer(answer, it.lang)
            rows[j] = {
                "task": it.task, "lang": it.lang, "id": it.meta.get("id"),
                "pred": pred, "gold": it.gold,
                "correct": bool(is_correct(pred, it.gold)),
                "output_script": script_dom(answer),
                "output_lang": detlang(answer),                       # response fidelity (Zhao rep_fidelity)
                "reasoning_lang": detlang(reasoning) if has_trace else "",  # reasoning fidelity (rea_fidelity)
                "n_new_tokens": n_new, "truncated": truncated, "max_new_tokens": int(max_new_tokens),
                "output": (full if keep_trace else answer),
            }
        if bi % 8 == 0:
            _mps_empty_cache()
    return [r for r in rows if r is not None]


@torch.no_grad()
def _run_likelihood(model, tok, items, *, hooks, device, mask_holder=None) -> list[dict]:
    rows: list[dict] = []
    for i, it in enumerate(items):
        if i and i % 16 == 0:
            _mps_empty_cache()   # each item is a distinct [ctx+cand] shape; clear so the MPS
                                 # shape cache can't balloon over a long likelihood dev sweep
        # MC letters are scored with a leading space so they tokenize as their own token
        conts = [f" {c}" for c in it.choices] if it.task == "mmlu" else list(it.choices)
        logprobs = score_continuations(
            model, tok, it.prompt, conts, hooks=hooks, device=device, mask_holder=mask_holder
        )
        best = int(np.argmax(logprobs))
        if it.task == "mmlu":
            pred = it.choices[best]                        # "A".."D"
            correct = is_correct_mc(pred, it.gold)
        else:                                              # xwinograd 2-way cloze
            pred = score_cloze((logprobs[0], logprobs[1]))
            correct = pred == it.gold
        rows.append({
            "task": it.task, "lang": it.lang, "id": it.meta.get("id"),
            "pred": str(pred), "gold": it.gold, "correct": bool(correct),
            "output_script": "", "output_lang": "",
            "logprobs": [round(x, 4) for x in logprobs],
        })
    return rows


_DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("E5_MAX_NEW_TOKENS", "512"))


def run_eval(
    model, tok, items, *, hooks=None, batch_size: int | None = None, max_new_tokens: int | None = None,
    device=None, max_batch_tokens: int | None = None, model_name: str | None = None,
    mask_holder=None, cond: str | None = None,
    spec=None, enable_thinking: bool | None = None, gen=None, intervention_spec=None,
) -> pd.DataFrame:
    """Evaluate `items` (any mix of tasks/languages) into one tidy per-item `DataFrame`.

    `hooks` is the intervention seam: `None` = clean baseline (P1); P2 passes
    `build_layer_hooks(spec)`. `mask_holder` (an `intervene.MaskHolder`, built from the same spec)
    drives prompt-only position scoping on the likelihood path. `cond` overrides the condition
    label stamped on every row (defaults to `"clean"` / `"intervention"`), so paired before/after
    rows can be tagged e.g. `"alg1_tuned"` / `"alg1_paperfixed"`. One row per item: `model, task,
    lang, id, pred, gold, correct, output_script, output_lang, cond`.

    Generation batching: explicit `batch_size`/`max_batch_tokens` args win; otherwise the
    `E5_EVAL_BATCH` / `E5_EVAL_MAX_BATCH_TOKENS` env vars (set by scripts/e5_scaleout.py for the
    big-VRAM boxes); otherwise the local-box defaults 16 / 4096. No call site in the repo passes
    them, so the env is the single knob for every search/eval in a driver run.

    Engine (`E5_ENGINE`, default "hf"): "vllm" routes the generative MGSM path to a warm, process-
    wide vLLM engine (continuous batching — much faster on floored languages where HF's straggler
    batching idles most of the batch). On that path the HF `model` is ignored; the intervention is
    applied from `intervention_spec` (an `InterventionSpec`/`AffineSpec`, the same object the search
    freezes) via `VllmRunner.interventions`, not the HF forward `hooks`. Pass `intervention_spec`
    alongside `hooks` — both are built from one spec, so the HF path (hooks) and the vLLM path
    (spec) stay in lock-step. The output DataFrame schema is identical across engines; only the
    generation kernel differs (greedy decoding diverges in the last bits across engines — the
    scripts/validate_vllm_* first-token gate is what established semantic equivalence).
    """
    if batch_size is None:
        batch_size = int(os.environ.get("E5_EVAL_BATCH", "16"))
    if max_batch_tokens is None:
        _mbt = int(os.environ.get("E5_EVAL_MAX_BATCH_TOKENS", "4096"))
        max_batch_tokens = _mbt if _mbt > 0 else None            # 0 = uncapped (prompts-only cap)
    device = device or get_device()
    tag = _model_tag(model, model_name)
    cond = cond or ("clean" if not hooks else "intervention")

    # Per-model inference policy (reasoning => thinking-ON + sampling + high cap + eos). Explicit
    # args win over the resolved spec; an unregistered tag => ModelSpec() defaults => today's exact
    # greedy / thinking-OFF / 512-cap behavior (backward-compat invariant).
    spec = spec if spec is not None else model_spec(tag)
    enable_thinking = spec.is_reasoning if enable_thinking is None else enable_thinking
    gen = spec.gen if gen is None else gen
    if max_new_tokens is None:
        max_new_tokens = spec.max_new_tokens if spec.max_new_tokens is not None else _DEFAULT_MAX_NEW_TOKENS

    gen_items = [it for it in items if it.task in _GENERATIVE]
    lik_items = [it for it in items if it.task in _LIKELIHOOD]
    unknown = {it.task for it in items} - _GENERATIVE - _LIKELIHOOD
    if unknown:
        raise ValueError(f"no eval path for task(s) {unknown}")

    rows: list[dict] = []
    gen_frame = None                                     # set only on the vLLM generative path
    if gen_items and _use_vllm():
        # Fail LOUD, not silent: the vLLM engine applies edits from `intervention_spec`, not from
        # HF forward `hooks`. A caller that passed real hooks but no spec (e.g. phase3's bespoke
        # oracle-transport hooks, which have no InterventionSpec form) would otherwise get CLEAN
        # generation silently mislabeled as intervened — exactly the kind of drift the vLLM gate
        # exists to prevent. Clean evals (hooks=None) and spec-carrying evals are unaffected.
        if intervention_spec is None and hooks:
            raise ValueError(
                "E5_ENGINE=vllm: this run_eval call passed HF forward `hooks` but no "
                "`intervention_spec`; the vLLM engine can't apply raw hooks, so the "
                "intervention would vanish silently. Pass intervention_spec=<the frozen "
                "InterventionSpec/AffineSpec>, or run this path with E5_ENGINE=hf (non-spec "
                "hooks such as phase3 oracle transport are unsupported on vLLM).")
        # vLLM generative path: reuse the warm process-wide engine singleton (built once, keyed by
        # model name). The HF `model` object is ignored here — the intervention is applied per-spec
        # inside VllmRunner.interventions(intervention_spec), and run_eval_vllm reuses
        # build_chat_inputs + the same scoring helpers, so the returned df's schema (columns +
        # order) matches _run_generative exactly. hooks/mask_holder (HF-model forward hooks) do not
        # apply on this engine and are ignored.
        from vllm_backend import run_eval_vllm
        full_name = (model_name
                     or getattr(getattr(model, "config", None), "_name_or_path", None) or tag)
        runner = _get_vllm_runner(full_name)
        gen_frame = run_eval_vllm(
            runner, tok, gen_items, spec=intervention_spec, cond=cond,
            max_new_tokens=max_new_tokens, model_name=full_name,
            enable_thinking=enable_thinking, gen=gen,
        )
    elif gen_items:
        rows += _run_generative(
            model, tok, gen_items, hooks=hooks, batch_size=batch_size,
            max_new_tokens=max_new_tokens, device=device, max_batch_tokens=max_batch_tokens,
            enable_thinking=enable_thinking, gen=gen, eos_token_id=spec.eos_token_id,
        )
    if lik_items:
        rows += _run_likelihood(
            model, tok, lik_items, hooks=hooks, device=device, mask_holder=mask_holder
        )

    if rows:                                             # HF-gen and/or likelihood rows
        df = pd.DataFrame(rows)
        df.insert(0, "model", tag)
        df["cond"] = cond
        df["engine"] = "hf"      # open.md Q1: engine provenance, stamped at the source. A mixed
                                 # run (vLLM gen + HF likelihood) keeps each source's own label.
        # vLLM gen (already model/cond-stamped) precedes any likelihood rows, matching the HF-path
        # order (generative rows first, then likelihood); columns align by name under concat.
        df = pd.concat([gen_frame, df], ignore_index=True) if gen_frame is not None else df
    elif gen_frame is not None:                          # vLLM generative only (the E7/E9 MGSM case)
        df = gen_frame
    else:                                                # no items: preserve the empty-df shape
        df = pd.DataFrame(rows)
        df.insert(0, "model", tag)
        df["cond"] = cond
        df["engine"] = "vllm" if _use_vllm() else "hf"
    _mps_empty_cache()   # release the MPS shape-keyed cache so reserved memory can't ratchet across
                         # the many run_eval calls a dev-split lever search makes (tuning.py)
    return df


def aggregate(df: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-(model, task, language) mean accuracy + language fidelities, with a bootstrap CI on acc.

    `response_fidelity` = fraction of outputs whose *answer* is in the target language (Zhao
    `rep_fidelity`; the Table-1 parenthetical). `reasoning_fidelity` = same for the `<think>` trace
    (Zhao `rea_fidelity`) — present **only** for reasoning-model rows that emitted a trace; NaN
    otherwise (matches Zhao having no reasoning-fidelity curve for non-reasoning models).
    """
    out = []
    keys = ["model", "task", "lang"]
    for (model, task, lang), g in df.groupby(keys):
        mean, lo, hi = bootstrap_ci(g["correct"].astype(float).to_numpy(), n_boot=n_boot, seed=seed)
        rec = {
            "model": model, "task": task, "lang": lang, "n": len(g),
            "acc": mean, "acc_lo": lo, "acc_hi": hi,
            "cond": g["cond"].iloc[0],
        }
        if "output_lang" in g.columns:                      # response fidelity (generative rows)
            genrows = g[g["output_lang"].astype(str) != ""]  # "" = likelihood rows, not generative
            if len(genrows):
                rec["undetectable_rate"] = float((genrows["output_lang"].astype(str) == "?").mean())
                det = genrows[genrows["output_lang"].astype(str) != "?"]   # '?' excluded from fidelity
                if len(det):
                    fid = (det["output_lang"] == lang).astype(float).to_numpy()
                    (rec["response_fidelity"], rec["response_fidelity_lo"],
                     rec["response_fidelity_hi"]) = bootstrap_ci(fid, n_boot=n_boot, seed=seed)
        if "reasoning_lang" in g.columns:                   # reasoning fidelity (trace rows only)
            rl = g["reasoning_lang"]
            rea = g[rl.notna() & ~rl.astype(str).isin(["", "?"])]
            if len(rea):
                rfid = (rea["reasoning_lang"] == lang).astype(float).to_numpy()
                (rec["reasoning_fidelity"], rec["reasoning_fidelity_lo"],
                 rec["reasoning_fidelity_hi"]) = bootstrap_ci(rfid, n_boot=n_boot, seed=seed)
        if "truncated" in g.columns:                        # differential-truncation instrument (#8)
            tr = g[g["truncated"].notna()]
            if len(tr):
                rec["truncation_rate"] = float(tr["truncated"].astype(float).mean())
        out.append(rec)
    return pd.DataFrame(out).sort_values(keys).reset_index(drop=True)


# MGSM's low-resource eval languages (E5 §6 H3): the languages where the paper's DiffPCA edge is
# hypothesized to concentrate. Membership is by `lang` code, so it applies to any task carrying
# these languages.
LOW_RESOURCE = frozenset({"sw", "te", "th", "bn"})


def paired_delta(
    df: pd.DataFrame, cond_b: str, cond_a: str, *, n_boot: int = 1000, seed: int = 0,
    low_resource=LOW_RESOURCE,
) -> pd.DataFrame:
    """Per-(model, task, lang) paired mean-acc delta `cond_b − cond_a` on identical items.

    The H2/H3 statistic: unlike `aggregate` (which forms *unpaired* per-condition CIs), this
    inner-joins the two conditions on `["model", "task", "lang", "id"]`, forms the per-item
    `correct_b − correct_a` difference vector, and bootstraps *that* vector — so the CI reflects
    the paired comparison on the same problems (variance from item difficulty cancels).

    Raises if any (model, task, lang) group doesn't pair 1:1 on `id` across the two conditions, so
    a silent mis-pairing can't corrupt the headline delta. Emits tidy columns: `model, task, lang,
    n, acc_a, acc_b, delta, delta_lo, delta_hi, low_resource`.
    """
    a = df[df["cond"] == cond_a]
    b = df[df["cond"] == cond_b]
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"missing rows for cond_a={cond_a!r} ({len(a)}) or cond_b={cond_b!r} ({len(b)})")
    keys = ["model", "task", "lang"]
    out = []
    for grp, ga in a.groupby(keys):
        gb = b[(b[keys] == pd.Series(dict(zip(keys, grp)))).all(axis=1)]
        ids_a = ga.set_index("id")["correct"].astype(float)
        ids_b = gb.set_index("id")["correct"].astype(float)
        if ids_a.index.has_duplicates or ids_b.index.has_duplicates:
            raise ValueError(f"duplicate item ids within a condition for group {grp}")
        if set(ids_a.index) != set(ids_b.index):
            raise ValueError(
                f"items do not pair 1:1 for group {grp}: "
                f"{len(set(ids_a.index) ^ set(ids_b.index))} unmatched id(s)"
            )
        ids_b = ids_b.reindex(ids_a.index)                  # align on the shared item order
        diff = (ids_b - ids_a).to_numpy()
        delta, lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
        model, task, lang = grp
        out.append({
            "model": model, "task": task, "lang": lang, "n": len(diff),
            "acc_a": float(ids_a.mean()), "acc_b": float(ids_b.mean()),
            "delta": delta, "delta_lo": lo, "delta_hi": hi,
            "low_resource": lang in low_resource,
        })
    return pd.DataFrame(out).sort_values(keys).reset_index(drop=True)


def low_resource_summary(
    paired_df: pd.DataFrame, item_df: pd.DataFrame | None = None, *,
    cond_b: str | None = None, cond_a: str | None = None, n_boot: int = 1000, seed: int = 0,
    low_resource=LOW_RESOURCE,
) -> pd.DataFrame:
    """Collapse `paired_delta` output to one signed delta + CI per `(task, low_resource)` group (H3).

    H3 ("the DiffPCA edge concentrates in low-resource languages") is a single signed number with a
    CI per group. Two corrections over a naive pool:
      - **Group by `task`** (not language alone): MGSM's low-resource edge must not be diluted by
        M-MMLU / XWinograd rows for the same languages (they outnumber MGSM ~6:1 at full scale).
      - **Problem-level CLUSTER bootstrap** when the per-item `item_df` + `cond` labels are given:
        MGSM is parallel, so the L language rows of one problem (shared id suffix `<task>-<lang>-<i>`)
        are NOT independent — an iid pool narrows the CI by the design effect. We resample problems,
        not items. The naive iid CI is reported alongside (`*_iid`) to document the design effect.
    Falls back to bootstrapping the per-language `delta` values when `item_df` is absent.
    Returns: `task, low_resource, n_langs, delta, delta_lo, delta_hi, delta_lo_iid, delta_hi_iid`.
    """
    out = []
    for (task, flag), g in paired_df.groupby(["task", "low_resource"]):
        if item_df is not None and cond_b is not None and cond_a is not None:
            langs = set(g["lang"])
            sub = item_df[(item_df["task"] == task) & (item_df["lang"].isin(langs))]
            # paired_delta validates 1:1 id-alignment per group before we pool items across langs
            paired_delta(sub, cond_b, cond_a, n_boot=1, seed=seed, low_resource=low_resource)
            join_keys = ["model", "task", "lang", "id"]
            a = sub[sub["cond"] == cond_a].set_index(join_keys)["correct"].astype(float)
            b = sub[sub["cond"] == cond_b].set_index(join_keys)["correct"].astype(float)
            b = b.reindex(a.index)                          # pooled per-item diff across the group
            diffs = (b - a).to_numpy()
            # cluster on the shared problem index (the `<i>` suffix of `<task>-<lang>-<i>`), so a
            # problem's parallel translations resample together
            ids = a.index.get_level_values("id").astype(str)
            clusters = np.array([s.rsplit("-", 1)[-1] for s in ids])
            delta, lo, hi = bootstrap_ci_cluster(diffs, clusters, n_boot=n_boot, seed=seed)
            _, lo_i, hi_i = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)   # naive iid, for reference
        else:
            delta, lo, hi = bootstrap_ci(g["delta"].to_numpy(), n_boot=n_boot, seed=seed)
            lo_i, hi_i = lo, hi
        out.append({
            "task": task, "low_resource": bool(flag), "n_langs": len(g),
            "delta": delta, "delta_lo": lo, "delta_hi": hi,
            "delta_lo_iid": lo_i, "delta_hi_iid": hi_i,
        })
    return pd.DataFrame(out).sort_values(["task", "low_resource"]).reset_index(drop=True)


def iut_cross_verdict(ci1: tuple, ci2: tuple, *, mde: float = 0.0) -> str:
    """G4 verdict for a cross arm vs its two parents via an intersection-union test on paired CIs.

    `ci1`/`ci2` are `(lo, hi)` of the paired delta (cross − parent) vs each parent. The E7 gate
    must be CI-based, not point-estimate dominance (a noisy cross beats a parent ~half the time
    under the null, and there are 26 cross cells):
      - **compounds**   — both CIs strictly above 0 (`lo > 0`): beats BOTH parents (the IUT);
      - **interferes**  — either CI strictly below 0 (`hi < 0`);
      - **redundant**   — both CIs inside `±mde` (indistinguishable within the min. detectable effect);
      - **indeterminate** otherwise.
    Feed `mde` from the run's power analysis; multiplicity across the ladder is handled by the caller
    (Holm across the 26 cells, or claiming 'compounds' only on ≥k pre-named models).
    """
    lo1, hi1 = ci1
    lo2, hi2 = ci2
    if lo1 > 0 and lo2 > 0:
        return "compounds"
    if hi1 < 0 or hi2 < 0:
        return "interferes"
    if abs(lo1) <= mde and abs(hi1) <= mde and abs(lo2) <= mde and abs(hi2) <= mde:
        return "redundant"
    return "indeterminate"


def classify_crosses(paired_df: pd.DataFrame, crosses: dict, *, mde: float = 0.0) -> pd.DataFrame:
    """Per-(model, task, lang) G4 verdict for each cross cell from a `paired_delta` frame.

    `crosses` maps `cross_arm -> (parent1, parent2)`, e.g.
    `{"diffpca_oblique": ("diffpca", "leace"), "diffpca_wiener": ("diffpca", "wiener")}`. `paired_df`
    must carry the contrasts `"<cross>_tuned - <parent>_tuned"` (as e6_phase2 emits) with
    `delta_lo/delta_hi`. Returns columns: `model, task, lang, cross, verdict, d_vs_<p1>, d_vs_<p2>`.
    """
    keys = ["model", "task", "lang"]
    out = []
    for cross, (p1, p2) in crosses.items():
        c1 = paired_df[paired_df["contrast"] == f"{cross}_tuned - {p1}_tuned"].set_index(keys)
        c2 = paired_df[paired_df["contrast"] == f"{cross}_tuned - {p2}_tuned"].set_index(keys)
        for k in c1.index.intersection(c2.index):
            r1, r2 = c1.loc[k], c2.loc[k]
            verdict = iut_cross_verdict((r1["delta_lo"], r1["delta_hi"]),
                                        (r2["delta_lo"], r2["delta_hi"]), mde=mde)
            out.append({**dict(zip(keys, k)), "cross": cross, "verdict": verdict,
                        f"d_vs_{p1}": float(r1["delta"]), f"d_vs_{p2}": float(r2["delta"])})
    return pd.DataFrame(out)


def holm_adjust(pvalues, *, alpha: float = 0.05) -> pd.DataFrame:
    """Holm–Bonferroni step-down over a family of raw p-values `{key: p}` (the multiplicity control).

    Discharges the "Holm across the cells" promise in `iut_cross_verdict`: sort the m raw p-values
    ascending; the i-th smallest (i from 0) is rejected iff it *and every smaller one* clear their
    step-down thresholds ``alpha / (m - i)`` — so once one p fails, all larger p's are non-rejected
    too (step-down monotonicity). The reported `p_holm` is the monotone (running-max) adjusted
    p-value ``max_{j<=i} min(1, (m - j)·p_(j))``, directly comparable to `alpha`. Pure and
    dependency-free; `pvalues` may be a `{key: p}` dict or a list of `(key, p)` pairs. Returns columns
    `key, p, p_holm, reject`, sorted by ascending `p`. Empty input => an empty typed frame.
    """
    items = list(pvalues.items()) if isinstance(pvalues, dict) else list(pvalues)
    if not items:
        return pd.DataFrame(columns=["key", "p", "p_holm", "reject"])
    m = len(items)
    order = sorted(range(m), key=lambda i: items[i][1])
    out, running, still_rejecting = [], 0.0, True
    for rank, idx in enumerate(order):
        key, p = items[idx][0], float(items[idx][1])
        running = max(running, min(1.0, (m - rank) * p))     # enforce monotone adjusted p
        if still_rejecting and p <= alpha / (m - rank):
            reject = True
        else:
            reject, still_rejecting = False, False           # first failure closes the step-down
        out.append({"key": key, "p": p, "p_holm": running, "reject": reject})
    return pd.DataFrame(out)


def write_results(
    df: pd.DataFrame, agg: pd.DataFrame, model_tag: str, *, phase: str = "phase1_clean",
    results_dir=None,
) -> tuple:
    """Emit model-stamped tidy CSVs: per-item and aggregated. Returns the two paths.

    Routed to results/<experiment>/<model_tag>/ by `results_path`; pass an explicit
    `results_dir` to write flat somewhere else instead.
    """
    item_name = f"E5_{phase}_{model_tag}.csv"
    agg_name = f"E5_{phase}_agg_{model_tag}.csv"
    if results_dir is None:
        item_path = results_path(item_name, model_tag)
        agg_path = results_path(agg_name, model_tag)
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        item_path, agg_path = results_dir / item_name, results_dir / agg_name
    df.to_csv(item_path, index=False)
    agg.to_csv(agg_path, index=False)
    return item_path, agg_path
