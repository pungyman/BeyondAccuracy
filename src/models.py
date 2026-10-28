"""Per-model inference registry: reasoning flag, decoding policy, loading quirks, and the ladder.

The repo's eval/loading path is otherwise uniform (one `Config`, one `load_model_and_tokenizer`).
Real models are not uniform: reasoning models (Qwen3/GLM-Z1/R1-distill) need thinking-ON MGSM eval
with sampling; a few need `trust_remote_code` or an EOS fix; the 14B needs a 4-bit fit. This module
holds those per-model facts as a tag-keyed registry — the exact shape of `tuning.TROUGH_LAYER` — plus
a resolver whose default (`ModelSpec()`) reproduces today's behavior for any unregistered tag, so the
7 non-reasoning models and the test stubs are byte-identical to before.

Imports only `torch` + stdlib (no cycle: `extract.py`/`evaluate.py` import this).

Decoding is Zhao et al. (2505.15257)-faithful: their `main_generate_steering_task.py` defaults to
`temperature=0` (greedy), and only their reasoning launch script overrides to `temperature 0.6 /
top_p 0.95 / max_tokens 16384`. So: greedy for non-reasoning, sampling for reasoning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

GEN_SEED = 0                 # fixed generation seed -> reproducible paired deltas (Zhao doesn't fix one)
_REASON_CAP = 16384          # reasoning token ceiling (matches Zhao --max_tokens 16384); EOS ends earlier


@dataclass(frozen=True)
class GenConfig:
    """Decoding policy. `do_sample=False` => greedy (only `pad_token_id` is passed, as today)."""
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None   # vLLM path only (run_eval_vllm passes it through)
    seed: int = GEN_SEED


# Shared decoding profiles (from the model cards / Zhao's reasoning script).
_GREEDY = GenConfig()                                                          # non-reasoning
_QWEN3 = GenConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)      # Qwen3 thinking mode
_DEEPSEEK = GenConfig(do_sample=True, temperature=0.6, top_p=0.95)            # R1-distill / GLM-Z1 (no top_k)


@dataclass(frozen=True)
class ModelSpec:
    """Per-model inference facts. Defaults reproduce today's uniform behavior exactly."""
    is_reasoning: bool = False                 # thinking-ON MGSM eval + </think> strip + reasoning cap
    trust_remote_code: bool = False            # custom modeling (internlm)
    dtype: torch.dtype | None = None           # None -> cfg.dtype (bf16 today)
    eos_token_id: int | list[int] | None = None  # override template/gen_config mismatch (Yi)
    max_new_tokens: int | None = None          # None -> _DEFAULT_MAX_NEW_TOKENS (512)
    gen: GenConfig = _GREEDY                    # decoding policy (greedy by default)
    attn_implementation: str | None = None
    n_layers: int | None = None                # metadata (smoke assertion / logging)
    d_model: int | None = None
    family: str | None = None
    massive_ratio: int | None = None           # max-activation / median ratio (angular guardrail)
    angular: str = "default"                    # "forced" iff massive_ratio >= ~1000 (screen hint)
    fit: str = "single"                         # "single" | "quant4bit" | "shard2" (14B)


# --- The 13-model ladder --------------------------------------------------------------------------
# HF ids, in tiers. The reasoning/angular/cap facts live in MODEL_REGISTRY below (keyed on short tag).
NON_THINKING_7 = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
)
REASONING_6 = (
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "THUDM/GLM-Z1-9B-0414",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
)
LADDER_13 = NON_THINKING_7 + REASONING_6

TIERS = {"non-thinking": NON_THINKING_7, "reasoning": REASONING_6, "all": LADDER_13}


# --- Per-model overrides (keyed on the short tag `name.split("/")[-1]`) ----------------------------
# Only tags needing non-default behavior must appear; the 7 non-reasoning models are listed for
# metadata (n_layers/d/angular) but resolve to defaults for behavior regardless.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- non-reasoning (metadata only; behavior = defaults) ---
    "Qwen2.5-1.5B-Instruct": ModelSpec(family="Qwen2.5", n_layers=28, d_model=1536),
    "Qwen2.5-3B-Instruct": ModelSpec(family="Qwen2.5", n_layers=36, d_model=2048),
    "Qwen2.5-7B-Instruct": ModelSpec(family="Qwen2.5", n_layers=28, d_model=3584,
                                     massive_ratio=1000, angular="forced"),
    "Llama-3.1-8B-Instruct": ModelSpec(family="Llama", n_layers=32, d_model=4096, massive_ratio=86),
    "Mistral-7B-Instruct-v0.3": ModelSpec(family="Mistral", n_layers=32, d_model=4096),
    "Phi-3.5-mini-instruct": ModelSpec(family="Phi", n_layers=32, d_model=3072),
    "gemma-2-2b-it": ModelSpec(family="Gemma", n_layers=26, d_model=2304),
    # Not in LADDER_13 by design — that tuple is the fixed E5/E7 sweep set and `TIERS["all"]`
    # means it. A registry entry alone is what a run needs; `model_spec` never reads the tiers.
    # Multimodal repo (`Gemma3ForConditionalGeneration`): the vision tower rides along and the
    # decoder blocks sit under the text tower — see `intervene._HF_LAYER_PATHS`. Behavior is all
    # defaults (greedy, bf16, 512-token cap, non-reasoning). `massive_ratio`/`angular` are left
    # unset deliberately: they are measured by `scripts/e4_phase0_geometry.py`, not guessed, and
    # an unset `angular` resolves to "default" rather than silently forcing the angular variant.
    "gemma-3-12b-it": ModelSpec(family="Gemma3", n_layers=48, d_model=3840),
    # --- reasoning ---
    "Qwen3-1.7B": ModelSpec(is_reasoning=True, gen=_QWEN3, max_new_tokens=_REASON_CAP,
                            family="Qwen3", n_layers=28, d_model=2048),
    "Qwen3-4B": ModelSpec(is_reasoning=True, gen=_QWEN3, max_new_tokens=_REASON_CAP,
                          family="Qwen3", n_layers=36, d_model=2560),
    "Qwen3-8B": ModelSpec(is_reasoning=True, gen=_QWEN3, max_new_tokens=_REASON_CAP,
                          family="Qwen3", n_layers=36, d_model=4096,
                          massive_ratio=1000, angular="forced"),
    "GLM-Z1-9B-0414": ModelSpec(is_reasoning=True, gen=_DEEPSEEK, max_new_tokens=_REASON_CAP,
                                family="GLM", n_layers=40, d_model=4096, massive_ratio=345),
    "DeepSeek-R1-Distill-Qwen-7B": ModelSpec(is_reasoning=True, gen=_DEEPSEEK, max_new_tokens=_REASON_CAP,
                                             family="R1-distill", n_layers=28, d_model=3584,
                                             massive_ratio=1000, angular="forced"),
    "DeepSeek-R1-Distill-Qwen-14B": ModelSpec(is_reasoning=True, gen=_DEEPSEEK, max_new_tokens=_REASON_CAP,
                                              family="R1-distill", n_layers=48, d_model=5120,
                                              massive_ratio=1000, angular="forced"),  # bf16 native: fits 96 GB (E7/GB202)
    # --- audit gotchas (not in the ladder; here so a stray run is safe) ---
    "Yi-1.5-9B-Chat": ModelSpec(eos_token_id=7, family="Yi", n_layers=48, d_model=4096),  # <|im_end|>=7
    "internlm2_5-7b-chat": ModelSpec(trust_remote_code=True, family="InternLM"),
}


def model_spec(model_name_or_tag: str) -> ModelSpec:
    """Resolve a `ModelSpec` from an HF id or short tag; unknown tags -> defaults (today's behavior).

    Env overrides (byte-identical when unset): `E5_ENABLE_THINKING` (0/1) forces the reasoning flag
    globally; `E5_REASON_MAX_NEW_TOKENS` overrides the reasoning token cap (the §6 speed lever).
    """
    tag = str(model_name_or_tag).split("/")[-1]
    spec = MODEL_REGISTRY.get(tag, ModelSpec())

    force = os.environ.get("E5_ENABLE_THINKING")
    if force is not None:
        spec = replace(spec, is_reasoning=(force not in ("", "0", "false", "False")))

    cap = os.environ.get("E5_REASON_MAX_NEW_TOKENS")
    if cap and spec.is_reasoning:
        spec = replace(spec, max_new_tokens=int(cap))

    return spec


def angular_forced(model_name_or_tag) -> bool:
    """True iff massive activations force the angular DiffPCA fit (raw DiffPCA collapses there).

    The 4 forced tags (Qwen2.5-7B, Qwen3-8B, DeepSeek-R1-Distill-7B/14B) carry `angular="forced"`;
    every other tag — registered or not — resolves to `angular="default"`. Drivers gate the raw
    (non-angular) DiffPCA sibling arm on `not angular_forced(tag)`: it runs as a fully-tuned arm on
    the 9 non-forced models, and is *skipped* on the forced ones — raw collapsing there is the
    finding, not a bug. A pure predicate, so `models.py` stays torch+stdlib-only.
    """
    return model_spec(model_name_or_tag).angular == "forced"
