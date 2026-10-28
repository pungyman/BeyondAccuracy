from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from config import LINALG_DEVICE

def language_means(acts: dict[str, Tensor]) -> Tensor:
    """Stack per-language mean activations into W = [m_1 | ... | m_L]  (d x L)."""
    cols = [a.to(LINALG_DEVICE, torch.float32).mean(dim=0) for a in acts.values()]
    # Column order is irrelevant to the subspace: permuting languages is W -> W @ Pi,
    # which leaves the LEFT singular vectors (M_s) and the anchor wc_new invariant (only
    # the right factors / Gamma permute). So M_s and P = M_s M_s^T don't depend on lang
    # order. Order only matters for *indexing* (e.g. heavy_index, mapping cols->langs),
    # so just keep it consistent (cfg.langs) across the pipeline. Verified numerically.
    return torch.stack(cols, dim=1)

def build_language_subspace(W: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Two-step SVD (mirrors mlrs build_aligner exactly). Returns (M_a, M_s). fp32/CPU."""
    W = W.to(LINALG_DEVICE, torch.float32)
    d, Ln = W.shape
    one = torch.ones(Ln, dtype=W.dtype, device=W.device)
    wc = W @ one / Ln
    centered = W - torch.outer(wc, one)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    Ws = U[:, :rank]
    Gamma = Vh.T[:, :rank] * S[:rank]
    best_fit = torch.outer(wc, one) + Ws @ Gamma.T
    wc_new = torch.linalg.pinv(best_fit).T @ one
    # Verbatim from the paper's build_aligner (see below). Dividing by the SQUARED norm
    # (not the norm) is not arbitrary: it is the unique scaling that makes the anchor
    # orthogonal to the resulting subspace. Since best_fit is full column rank (rank r+1=L
    # at default rank), wc_raw = (best_fit^+)^T 1 satisfies best_fit^T wc_raw = 1, so each
    # column has the same inner product with wc_raw. Rescaling by 1/||wc_raw||^2 forces
    # best_fit^T wc_new = ||wc_new||^2 * 1, hence prod^T wc_new = best_fit^T wc_new -
    # 1 (wc_new^T wc_new) = 0  =>  wc_new _|_ span(M_s). (Unit-norm scaling would NOT.)
    wc_new = wc_new / (wc_new ** 2).sum()
    prod = best_fit - torch.outer(wc_new, one)
    U2, S2, _ = torch.linalg.svd(prod, full_matrices=False)
    M_s = U2[:, :rank]
    return wc_new, M_s  # this is M_a and M_s from the paper

def projector(M_s: Tensor) -> Tensor:
    """Orthogonal projector onto span(M_s): P = M_s M_s^T (sign/rotation invariant)."""
    return M_s @ M_s.T

def build_aligner(rank: int, lan_emb):
    """REFERENCE numpy implementation, verbatim from mlrs/src/main_multilingual_space.py."""
    lan_mean_emb = {lan: np.mean(emb, axis=0) for lan, emb in lan_emb.items()}
    W = np.stack(list(lan_mean_emb.values())).T
    _, D = W.shape
    wc = W @ np.ones(D) / D
    u, s, vh = np.linalg.svd(W - wc.reshape(-1, 1) @ np.ones((1, D)))
    Ws, Gamma = u[:, :rank], vh.T[:, :rank] @ np.diag(s[:rank])
    best_fit_W = wc.reshape(-1, 1) @ np.ones((1, D)) + Ws @ Gamma.T
    wc_new = np.linalg.pinv(best_fit_W).T @ np.ones(D)
    wc_new /= (wc_new ** 2).sum()
    prod = best_fit_W - wc_new.reshape(-1, 1) @ np.ones((1, D))
    u, s, vh = np.linalg.svd(prod)
    ws_new = u[:, :rank]
    return wc_new, ws_new

def _anchored_subspace(W: Tensor, rank: int, weight: Tensor) -> tuple[Tensor, Tensor]:
    """Generalized two-step builder: project out a single anchor read by `weight`.

    Repo-original E1 helper (not in the paper). Identical to `build_language_subspace`
    when `weight == 1` (verified numerically); `weight` only changes which anchor is read
    off via the pseudo-inverse, while deflation still uses `one`.
    """
    W = W.to(LINALG_DEVICE, torch.float32)
    d, Ln = W.shape
    one = torch.ones(Ln, dtype=W.dtype, device=W.device)
    wc = W @ one / Ln
    centered = W - torch.outer(wc, one)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    Ws = U[:, :rank]; Gamma = Vh.T[:, :rank] * S[:rank]
    best_fit = torch.outer(wc, one) + Ws @ Gamma.T
    wc_new = torch.linalg.pinv(best_fit).T @ weight          # weight replaces `one`
    wc_new = wc_new / (wc_new ** 2).sum()
    prod = best_fit - torch.outer(wc_new, one)
    U2, _, _ = torch.linalg.svd(prod, full_matrices=False)
    return wc_new, U2[:, :rank]

def build_subspace_variant(W: Tensor, rank: int, mode: str = "uniform",
                           heavy_index: int = 0, heavy_weight: float = 3.0):
    """E1 anchor variants. Returns (M_s (d,effective_rank), meta dict).

    Repo-original E1 experiments (not in the paper). `mode="uniform"` reproduces
    `build_language_subspace` exactly; see docs/modules/subspace.md for the mode table.
    """
    W = W.to(LINALG_DEVICE, torch.float32)
    d, Ln = W.shape
    one = torch.ones(Ln, dtype=W.dtype, device=W.device)
    if mode == "uniform":
        _, M_s = _anchored_subspace(W, rank, one)
        return M_s, {"effective_rank": rank, "mode": mode}
    if mode == "weighted":
        w = one.clone(); w[heavy_index] = heavy_weight          # concentrate on a non-English lang
        _, M_s = _anchored_subspace(W, rank, w)
        return M_s, {"effective_rank": rank, "mode": mode, "heavy_index": heavy_index}
    if mode == "rank3":
        # rank-3 SHARED (language-agnostic) subspace = top-3 left singular vectors of the
        # UNCENTERED mean matrix W (dominated by the common centroid). Deflate it from W;
        # M_s is the residual language variation, rank L-3 (2 fewer than the rank-1 anchor).
        eff = rank - 2
        Ub, _, _ = torch.linalg.svd(W, full_matrices=False)
        shared = Ub[:, :3]
        deflated = W - shared @ (shared.T @ W)
        U2, _, _ = torch.linalg.svd(deflated, full_matrices=False)
        return U2[:, :eff], {"effective_rank": eff, "mode": mode, "shared_rank": 3}
    if mode == "none":
        wc = W @ one / Ln
        U, _, _ = torch.linalg.svd(W - torch.outer(wc, one), full_matrices=False)
        return U[:, :rank], {"effective_rank": rank, "mode": mode}
    raise ValueError(f"unknown mode {mode!r}")
