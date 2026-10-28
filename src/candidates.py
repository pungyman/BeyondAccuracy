"""The five comparable rank-r subspaces at a single layer (see docs/experiments/shared_methods.md
sections 3-4). One source of truth reused by E3 (residual sweep) and E4 (geometry, frontier).

All five are built from the *same* per-language activations at one layer, so they are directly
comparable via `geometry.projector_distance` and downstream interventions:

    alg1     two-step SVD, refined anchor          (subspace.build_language_subspace)
    none     PCA of centered means                 (subspace.build_subspace_variant mode="none")
    diffpca  PCA of within-problem residuals        (this module)
    pooled   top-variance / content directions      (controls.pooled_pca_subspace)
    random   Haar-random rank-r                      (controls.random_orthonormal_subspace)
"""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from sklearn.decomposition import PCA

from config import LINALG_DEVICE
from subspace import language_means, build_language_subspace, build_subspace_variant
from controls import pooled_pca_subspace, random_orthonormal_subspace
from probe import residualize_problem, unit_rows

CANDIDATE_NAMES: tuple[str, ...] = ("alg1", "none", "diffpca", "pooled", "random")


def pooled_long_form(acts_by_lang: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Stack per-language `(n, d)` activations into pooled long form.

    Returns `(X, lang_idx, problem_idx)` with `X` of shape `(nL, d)` in **language-major**
    order (all of language 0, then language 1, ...). `problem_idx` therefore tiles `0..n-1`
    once per language, which is exactly the grouping `probe.residualize_problem` partials out
    (each problem's L parallel translations share a `problem_idx`).
    """
    langs = list(acts_by_lang)
    mats = [acts_by_lang[l].to(LINALG_DEVICE, torch.float32) for l in langs]
    n = mats[0].shape[0]
    L = len(langs)
    assert all(m.shape[0] == n for m in mats), "non-parallel: unequal n across languages"
    X = torch.cat(mats, dim=0)                              # (nL, d)
    lang_idx = torch.arange(L).repeat_interleave(n)         # [0]*n + [1]*n + ...
    problem_idx = torch.arange(n).repeat(L)                 # [0..n-1] tiled L times
    return X, lang_idx, problem_idx


def diffpca_subspace(
    acts_by_lang: dict[str, Tensor], rank: int, *, angular: bool = True
) -> tuple[Tensor, np.ndarray]:
    """DiffPCA: top-`rank` PCs of the within-problem-residualized activations.

    Equivalent to the metric-optimal subspace of shared_methods section 4.3. Returns
    `(M_s, explained_variance_ratio_)` where `M_s` is `(d, rank)` orthonormal and the scree is
    a length-`rank` array (its tail is informative for the intrinsic-dimensionality question).

    `angular` selects the PCA objective on the residual rows `R`:

    - `angular=True` (default, **shipped**): fit on **L2-normalized** rows (`probe.unit_rows`),
      so the objective matches the per-vector, scale-free captured fraction rho the sweep reports
      rather than norm-weighted variance. This is the massive-activation guardrail: without it,
      models with massive-activation dimensions (Qwen 7B+: a couple of coordinates ~1000x the
      typical scale) give heavy-tailed difference-vector norms and raw PCA spends its rank budget
      on a few giant-norm outliers -- so DiffPCA, the supposed per-prompt *ceiling*, drops *below*
      the mean-based subspaces. The scree is then the explained-variance ratio of the normalized
      residuals (scale-free, comparable across models); its tail still tracks intrinsic dim.
    - `angular=False`: fit on the **raw** residuals (norm-weighted variance PCA). Kept as an
      explicit option for the E4 Phase-0 angular-vs-raw geometry reconciliation and for
      massive-activation-free models where the two coincide; do **not** use it as the default on a
      flagged model (see E3_results §methods addendum).
    """
    X, _, problem_idx = pooled_long_form(acts_by_lang)
    R = residualize_problem(X, problem_idx)                 # np.float64 (nL, d), content removed
    fit_rows = unit_rows(R) if angular else R               # angular=outlier-robust; raw=norm-weighted
    pca = PCA(n_components=rank).fit(fit_rows)
    M_s = torch.tensor(pca.components_.T, dtype=torch.float32)   # (d, rank)
    return M_s, pca.explained_variance_ratio_


def build_candidate_subspaces(
    acts_by_lang: dict[str, Tensor],
    rank: int,
    seed: int = 0,
    include: tuple[str, ...] = CANDIDATE_NAMES,
    diffpca_angular: bool = True,
) -> tuple[dict[str, Tensor], dict]:
    """Build the requested rank-`rank` subspaces from one layer's per-language activations.

    Returns `(subspaces, meta)` where `subspaces[name]` is `(d, rank)` orthonormal and `meta`
    carries side information (currently the DiffPCA scree under `meta["diffpca_scree"]`).

    `diffpca_angular` is forwarded to `diffpca_subspace` (default `True` = the shipped angular,
    massive-activation-robust fit; `False` = raw norm-weighted variance PCA).
    """
    W = language_means(acts_by_lang)                        # (d, L)
    d = W.shape[0]
    subs: dict[str, Tensor] = {}
    meta: dict = {}
    if "alg1" in include:
        subs["alg1"] = build_language_subspace(W, rank)[1]
    if "none" in include:
        subs["none"] = build_subspace_variant(W, rank, mode="none")[0]
    if "diffpca" in include:
        M_s, scree = diffpca_subspace(acts_by_lang, rank, angular=diffpca_angular)
        subs["diffpca"] = M_s
        meta["diffpca_scree"] = scree
    if "pooled" in include:
        subs["pooled"] = pooled_pca_subspace(acts_by_lang, rank)
    if "random" in include:
        subs["random"] = random_orthonormal_subspace(d, rank, seed)
    return subs, meta


def layer_slice(acts: dict[str, Tensor], layer: int) -> dict[str, Tensor]:
    """Slice a `{lang: (n, n_layers, d)}` activation cache to one layer's `{lang: (n, d)}`.

    This is the single-layer view `build_candidate_subspaces` / `diffpca_subspace` consume, so
    the per-layer builder below is just a loop over `build_candidate_subspaces(layer_slice(...))`.
    """
    return {l: a[:, layer, :] for l, a in acts.items()}


def build_layer_subspaces(
    acts: dict[str, Tensor],
    layers,
    *,
    name: str = "alg1",
    rank: int | None = None,
    seed: int = 0,
    diffpca_angular: bool = True,
) -> dict[int, Tensor]:
    """Per-layer rank-`rank` subspaces `{layer -> M_s (d, rank)}` for one candidate `name`.

    Slices the `(n, n_layers, d)` cache one layer at a time (`layer_slice`) and reuses the
    single-layer `build_candidate_subspaces`, so this is the *only* new construction step P2
    needs — it stays one source of truth with E3/E4. `rank` defaults to `L-1` (the Alg1 / `none`
    structural cap, `cfg.rank`); `diffpca_angular` / `rank` are forwarded unchanged so P3 reuses
    this verbatim for DiffPCA (`name="diffpca"`, arbitrary `rank`).
    """
    rank = rank if rank is not None else len(acts) - 1
    out: dict[int, Tensor] = {}
    for layer in layers:
        subs, _ = build_candidate_subspaces(
            layer_slice(acts, layer), rank=rank, seed=seed,
            include=(name,), diffpca_angular=diffpca_angular,
        )
        out[int(layer)] = subs[name]
    return out
