"""Geometry + statistics helpers for the residual / subspace-comparison experiments
(docs/experiments/shared_methods.md section 5). Pure linear algebra and resampling: no model,
no I/O, so these are cheap and unit-testable.

- vector-vs-subspace: `captured_fraction` (the E3 metric rho)
- subspace-vs-subspace: `principal_angles`, `projector_distance`, `subspace_overlap` (E4 Phase 0)
- bookkeeping: `split_problems`, `ordered_or_unordered_pairs`
- inference: `bootstrap_ci`
"""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor

from config import LINALG_DEVICE


def split_problems(n: int, frac: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint train/test split over problem indices ``[0, n)``.

    `frac` is the train fraction. Returns `(train_idx, test_idx)` as sorted int arrays; a
    problem's parallel translations never straddle the split because we split on the problem
    index itself (shared_methods section 7).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(frac * n))
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def ordered_or_unordered_pairs(langs, ordered: bool = False) -> list[tuple[str, str]]:
    """Language pairs. `ordered=False` -> C(L,2) unordered (55 for L=11); `ordered=True` ->
    L(L-1) ordered (110). Order is irrelevant for the symmetric rho metric, so default to
    unordered; keep ordered only for directed analyses (E3 section 7)."""
    langs = list(langs)
    pairs: list[tuple[str, str]] = []
    for i, s in enumerate(langs):
        for j, t in enumerate(langs):
            if i == j:
                continue
            if not ordered and j < i:
                continue
            pairs.append((s, t))
    return pairs


def captured_fraction(V: Tensor, M_s: Tensor) -> Tensor:
    """Captured-variance fraction rho(v) = ||P v||^2 / ||v||^2 for each row of `V`.

    `M_s` is `(d, r)` **orthonormal**, so ``||P v||^2 = ||M_s^T v||^2`` and no `(d, d)`
    projector is ever materialized. `V` is `(m, d)`; returns `(m,)` in `[0, 1]`.
    """
    V = V.to(LINALG_DEVICE, torch.float32)
    M_s = M_s.to(LINALG_DEVICE, torch.float32)
    num = (V @ M_s).pow(2).sum(dim=1)
    den = (V * V).sum(dim=1).clamp_min(1e-12)
    return num / den


def principal_angles(Ma: Tensor, Mb: Tensor) -> Tensor:
    """Principal angles (radians, ascending) between `span(Ma)` and `span(Mb)`.

    Both `(d, r)` orthonormal. cos(theta_i) are the singular values of ``Ma^T Mb``.
    """
    Ma = Ma.to(LINALG_DEVICE, torch.float32)
    Mb = Mb.to(LINALG_DEVICE, torch.float32)
    cos = torch.linalg.svdvals(Ma.T @ Mb).clamp(-1.0, 1.0)
    return torch.arccos(cos)


def projector_distance(Ma: Tensor, Mb: Tensor) -> float:
    """Basis-invariant subspace distance ``||P_a - P_b||_F = sqrt(2) * ||sin(Theta)||_2``."""
    theta = principal_angles(Ma, Mb)
    return float(torch.sqrt(2.0 * (torch.sin(theta) ** 2).sum()))


def subspace_overlap(Ma: Tensor, Mb: Tensor) -> float:
    """Captured-energy overlap ``tr(P_a P_b) / r = ||Ma^T Mb||_F^2 / r`` in `[0, 1]`
    (fraction of `b`'s basis captured by `a`; `r` = rank of `b`)."""
    Ma = Ma.to(LINALG_DEVICE, torch.float32)
    Mb = Mb.to(LINALG_DEVICE, torch.float32)
    r = Mb.shape[1]
    return float((Ma.T @ Mb).pow(2).sum() / r)


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0, ci: float = 0.95):
    """Percentile bootstrap CI for the mean, resampling the leading axis (problems).

    `values` is 1-D (e.g. per-problem mean rho). Returns `(mean, lo, hi)`.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return float(x.mean()), lo, hi


def bootstrap_ci_cluster(values, clusters, n_boot: int = 1000, seed: int = 0, ci: float = 0.95):
    """Cluster (block) bootstrap CI for the mean: resample whole CLUSTERS with replacement.

    For correlated designs the iid `bootstrap_ci` undercounts variance by the design effect
    ``1 + (m-1)ρ``. The canonical case here: MGSM is *parallel*, so the L language rows of one
    problem share its difficulty — pooling them as iid items narrows the CI spuriously. `clusters[i]`
    is the cluster label of `values[i]` (e.g. the shared problem index); we resample the K unique
    labels with replacement and pool every member row of each drawn cluster. Returns `(mean, lo, hi)`
    — the point estimate is the plain item mean; only the CI widens.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    c = np.asarray(clusters).ravel()
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    _, inv = np.unique(c, return_inverse=True)
    K = int(inv.max()) + 1
    members = [np.where(inv == k)[0] for k in range(K)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, K, K)                       # K clusters, with replacement
        boots[b] = x[np.concatenate([members[k] for k in pick])].mean()
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return float(x.mean()), lo, hi


def bootstrap_pvalue_cluster(values, clusters, *, side="greater", n_boot=1000, seed=0) -> float:
    """One-sided cluster-bootstrap p-value for ``mean(values)`` vs 0, resampling whole CLUSTERS.

    The p-value companion to `bootstrap_ci_cluster`: it draws from the *same* problem-cluster
    resample, so a Holm family built on these p's is consistent with the CIs reported beside it
    (E7 headline-validity package). `clusters[i]` labels `values[i]` (e.g. the shared problem index);
    we resample the K unique labels with replacement, pool every member row of each drawn cluster,
    and invert the resulting bootstrap distribution of the mean into a tail probability against 0:

      - ``side="greater"`` (H1: mean > 0): ``p = (1 + #{boot_mean <= 0}) / (n_boot + 1)``
      - ``side="less"``    (H1: mean < 0): ``p = (1 + #{boot_mean >= 0}) / (n_boot + 1)``

    The ``+1`` is the standard Davison–Hinkley finite-sample correction (p is never exactly 0). An
    all-positive vector => p ≈ 0; an all-negative vector => p ≈ 1. Returns a float in ``(0, 1]``;
    empty `values` returns NaN.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    c = np.asarray(clusters).ravel()
    n = len(x)
    if n == 0:
        return float("nan")
    _, inv = np.unique(c, return_inverse=True)
    K = int(inv.max()) + 1
    members = [np.where(inv == k)[0] for k in range(K)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, K, K)                       # K clusters, with replacement
        boots[b] = x[np.concatenate([members[k] for k in pick])].mean()
    if side == "greater":
        hits = int(np.sum(boots <= 0.0))
    elif side == "less":
        hits = int(np.sum(boots >= 0.0))
    else:
        raise ValueError(f"side must be 'greater' or 'less', got {side!r}")
    return (1.0 + hits) / (n_boot + 1.0)
