from __future__ import annotations
import contextlib
from dataclasses import dataclass
import torch
from torch import Tensor

def ablate(h: Tensor, M_s: Tensor, lam: float = 1.0) -> Tensor:
    """Remove language-subspace component: h - lam * (h @ M_s) @ M_s.T.

    lam scales the ablation extent: 0 is identity, 1 full removal, >1 over-ablation.
    lam < 0 *re-injects* (amplifies the subspace component) — the high-layer band of E5's
    mid-ablate / high-reinject schedule (benchmarks.md §5).
    """
    M_s = M_s.to(device=h.device, dtype=h.dtype)
    return h - lam * (h @ M_s) @ M_s.T

def swap(h: Tensor, M_s: Tensor, mu_s: Tensor, mu_t: Tensor, alpha: float = 1.0) -> Tensor:
    """Eq. 3': strip source subspace coords, recenter source->target mean."""
    M_s = M_s.to(device=h.device, dtype=h.dtype)
    mu_s = mu_s.to(device=h.device, dtype=h.dtype)
    mu_t = mu_t.to(device=h.device, dtype=h.dtype)
    return ablate(h, M_s) - mu_s + alpha * mu_t

# Decoder-layer paths on the HF side, in probe order. `model.layers` covers every text-only
# causal LM in the ladder (Qwen, Llama, Mistral, Phi, Gemma 2). The `language_model` variants are
# multimodal wrappers — Gemma 3 12B/27B are `Gemma3ForConditionalGeneration`, a text tower plus a
# SigLIP vision tower, so the blocks sit one or two levels deeper than the text-only case. The
# exact depth moved between transformers releases, hence a list rather than a single path.
# Mirrors `vllm_backend.resolve_layers`, which does the same probe on the vLLM copy.
_HF_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.language_model.model.layers",
)


def _dig(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_hf_layers(model):
    """The decoder-layer `ModuleList` on an HF model, across text-only and multimodal layouts."""
    errors = []
    for path in _HF_LAYER_PATHS:
        try:
            layers = _dig(model, path)
        except AttributeError as e:  # noqa: PERF203 - probing distinct layouts
            errors.append(f"{path}: {e}")
            continue
        if hasattr(layers, "__len__") and len(layers):
            return layers
        errors.append(f"{path}: resolved but empty")
    raise RuntimeError(
        f"no decoder-layer list found on {type(model).__name__}; tried:\n  "
        + "\n  ".join(errors)
        + "\nInspect the model and add the layout to `_HF_LAYER_PATHS`."
    )


@contextlib.contextmanager
def interventions(model, hooks: dict[int, callable]):
    """Register residual-stream forward hooks on decoder blocks {layer_idx: fn(h)->h}.
    Applies on every forward (prompt + each generation step). Removes cleanly on exit."""
    handles = []
    layers = resolve_hf_layers(model)
    def _make(fn):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (fn(output[0]),) + tuple(output[1:])
            return fn(output)
        return hook
    try:
        for idx, fn in hooks.items():
            handles.append(layers[idx].register_forward_hook(_make(fn)))
        yield
    finally:
        for h in handles:
            h.remove()


# --- E5 P2: per-layer, scope-aware intervention wrapping `ablate` ------------------------------

@dataclass(frozen=True)
class InterventionSpec:
    """A frozen, per-layer intervention (benchmarks.md §5, hyperparameter_search.md lever 2).

    - `subspaces`: `{layer -> M_s (d, r)}`, one per layer, built from parallel MATH (E5 §2).
    - `lam`: `{layer -> strength}`. Sign encodes the two bands of the edit `h - lam·(h@M_s)@M_sᵀ`:
      `lam>0` ablates (mid layers, remove language), `lam<0` re-injects (high layers, keep output
      fluent). Layers absent from `lam` (or set to exactly 0) are left untouched.
    - `scope`: `"prompt_only"` (reported) edits only the prefill pass (generative) / context
      positions (likelihood); `"all_steps"` edits every decode step / all positions.
    """
    subspaces: dict[int, Tensor]
    lam: dict[int, float]
    scope: str = "prompt_only"


@dataclass
class MaskHolder:
    """Mutable per-forward position mask for the likelihood scope (set by the harness).

    The likelihood path has no decode loop, so `prompt_only` vs `all_steps` becomes a position
    mask over the single `[context + candidate]` forward. The harness calls `set_from_ctx(...)`
    before each teacher-forced forward; every layer hook reads `.mask` and edits only the True
    positions. `scope` lets the harness build the right mask from a context length without
    re-plumbing it. `.mask` is `None` outside a masked forward (e.g. during generation), which is
    the signal the generative hook branch uses to fall back to prefill detection.
    """
    scope: str = "prompt_only"
    mask: Tensor | None = None

    def set_from_ctx(self, seq_len: int, ctx_len: int) -> None:
        self.mask = edit_mask(seq_len, ctx_len, self.scope)

    def clear(self) -> None:
        self.mask = None


def edit_mask(seq_len: int, ctx_len: int, scope: str = "prompt_only") -> Tensor:
    """Boolean `(seq_len,)` mask of positions to edit in a likelihood `[context+candidate]` forward.

    `prompt_only` -> True over the `ctx_len` context positions only (the candidate tokens, i.e. the
    scored continuation, pass through untouched); `all_steps` -> True everywhere.
    """
    m = torch.zeros(seq_len, dtype=torch.bool)
    if scope == "all_steps":
        m[:] = True
    else:
        m[:min(ctx_len, seq_len)] = True
    return m


def _apply_masked(h: Tensor, M_s: Tensor, lam: float, mask: Tensor) -> Tensor:
    """Ablate only the `mask`-True positions of `h` (`(..., seq, d)`), leave the rest unchanged."""
    edited = ablate(h, M_s, lam)
    m = mask.to(device=h.device).view(*([1] * (h.dim() - 2)), -1, 1)
    return torch.where(m, edited, h)


def _layer_fn(M_s: Tensor, lam: float, scope: str, mask_holder: "MaskHolder | None"):
    def fn(h: Tensor) -> Tensor:
        mask = mask_holder.mask if mask_holder is not None else None
        if mask is not None:                       # likelihood path: scope lives in the mask
            return _apply_masked(h, M_s, lam, mask)
        # generative path: prompt_only edits the prefill (seq>1) only; decode steps (seq==1) pass
        if scope == "prompt_only" and h.shape[-2] == 1:
            return h
        return ablate(h, M_s, lam)
    return fn


def build_layer_hooks(spec: InterventionSpec, mask_holder: "MaskHolder | None" = None
                      ) -> dict[int, callable]:
    """Turn an `InterventionSpec` into `{layer: fn(h)->h}` for `interventions(model, hooks)`.

    Each `fn` applies `ablate(h, spec.subspaces[layer], spec.lam[layer])` with the scope logic
    above. Layers whose `lam` is exactly 0 are dropped (no-op ablation), so an ablate-only sweep
    (`lam_high=0`) naturally registers no re-injection hooks. Pass `mask_holder` to enable the
    likelihood position-masking path; leave it `None` for the generative / clean path. The return
    is a plain dict, so it drops straight into `run_eval(..., hooks=...)` (the P1 seam).
    """
    hooks: dict[int, callable] = {}
    for layer, lam in spec.lam.items():
        if lam == 0.0:
            continue
        hooks[int(layer)] = _layer_fn(spec.subspaces[layer], lam, spec.scope, mask_holder)
    return hooks


# --- E6: affine (non-projection) interventions --------------------------------------------------
#
# The E6 Plan-B operators (LEACE-style oblique erasure, Wiener shrinkage, moment-matching /
# paired-translator transports — operators.py) are all *affine* edits of the residual stream, a
# strict generalization of `ablate`'s hard orthogonal projection. `AffineEdit` stores one layer's
# edit in factored form so full-rank d x d matrices are never materialized in the hook path;
# `AffineSpec` / `build_affine_hooks` mirror `InterventionSpec` / `build_layer_hooks` exactly
# (same lam-sign band convention, same scope semantics, same MaskHolder plumbing) so the E5
# tuning/eval stack drives either spec unchanged.

@dataclass(frozen=True)
class AffineEdit:
    """One layer's affine edit in factored form.

    At strength `lam` the edit is `h <- h - lam * delta(h)` with

        delta(h) = diag * z + (z @ V) @ U.T + z @ dense.T + b,     z = h - c

    where every component is optional (`None` contributes nothing):

    - `c` `(d,)` — center the edit is measured from (typically the pooled train mean; `None` = 0).
    - `diag` `(d,)` — per-coordinate shrink (AdaIN moment matching, uniform-shrink control).
    - `U`, `V` `(d, r)` — low-rank factor pair: `delta` gains `(z @ V) @ U.T`. This is the LEACE
      oblique projection (`U = Sigma^{1/2} Q`, `V = Sigma^{-1/2} Q`), the Wiener gain, and the
      paired-translator correction; a plain subspace ablation is the special case `U = V = M_s`.
    - `dense` `(d, d)` — full matrix escape hatch (full-covariance transports at `rank=None`).
    - `b` `(d,)` — constant shift (mean-shift transport, the `mu_s - mu_t` term of swaps).

    `lam` keeps the `InterventionSpec` sign convention: 0 identity, 1 the full edit, negative
    re-injects (adds the removed component back — the high-layer band). Factors are stored on
    CPU/fp32 and cast to the activation's device/dtype per call, matching `ablate`.
    """
    c: Tensor | None = None
    diag: Tensor | None = None
    U: Tensor | None = None
    V: Tensor | None = None
    dense: Tensor | None = None
    b: Tensor | None = None

    def delta(self, h: Tensor) -> Tensor:
        """The component removed at `lam = 1`, shaped like `h` (`(..., d)`)."""
        z = h if self.c is None else h - self.c.to(device=h.device, dtype=h.dtype)
        out = torch.zeros_like(h)
        if self.diag is not None:
            out = out + z * self.diag.to(device=h.device, dtype=h.dtype)
        if self.U is not None:
            U = self.U.to(device=h.device, dtype=h.dtype)
            V = self.V.to(device=h.device, dtype=h.dtype)
            out = out + (z @ V) @ U.T
        if self.dense is not None:
            out = out + z @ self.dense.to(device=h.device, dtype=h.dtype).T
        if self.b is not None:
            out = out + self.b.to(device=h.device, dtype=h.dtype)
        return out

    def apply(self, h: Tensor, lam: float = 1.0) -> Tensor:
        return h - lam * self.delta(h)


@dataclass(frozen=True)
class AffineSpec:
    """`InterventionSpec` analogue for affine edits: `{layer -> AffineEdit}` + per-layer `lam`.

    Same band convention (`lam>0` remove at mid layers, `lam<0` re-inject at high layers, 0 or
    absent = untouched) and the same `scope` semantics as `InterventionSpec`.
    """
    edits: dict[int, AffineEdit]
    lam: dict[int, float]
    scope: str = "prompt_only"


def _apply_masked_affine(h: Tensor, edit: AffineEdit, lam: float, mask: Tensor) -> Tensor:
    """Affine-edit only the `mask`-True positions of `h` (`(..., seq, d)`)."""
    edited = edit.apply(h, lam)
    m = mask.to(device=h.device).view(*([1] * (h.dim() - 2)), -1, 1)
    return torch.where(m, edited, h)


def _affine_layer_fn(edit: AffineEdit, lam: float, scope: str, mask_holder: "MaskHolder | None"):
    def fn(h: Tensor) -> Tensor:
        mask = mask_holder.mask if mask_holder is not None else None
        if mask is not None:                       # likelihood path: scope lives in the mask
            return _apply_masked_affine(h, edit, lam, mask)
        if scope == "prompt_only" and h.shape[-2] == 1:
            return h
        return edit.apply(h, lam)
    return fn


def build_affine_hooks(spec: AffineSpec, mask_holder: "MaskHolder | None" = None
                       ) -> dict[int, callable]:
    """Turn an `AffineSpec` into `{layer: fn(h)->h}` — the affine twin of `build_layer_hooks`.

    Identical drop-zero / scope / mask semantics, so the E5 harness (`run_eval(..., hooks=...)`,
    `tuning` searches) is agnostic to which spec produced the hooks.
    """
    hooks: dict[int, callable] = {}
    for layer, lam in spec.lam.items():
        if lam == 0.0:
            continue
        hooks[int(layer)] = _affine_layer_fn(spec.edits[layer], lam, spec.scope, mask_holder)
    return hooks
