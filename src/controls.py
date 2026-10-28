from __future__ import annotations
import torch
from torch import Tensor
from config import LINALG_DEVICE
from subspace import language_means, build_language_subspace

def random_orthonormal_subspace(d: int, rank: int, seed: int) -> Tensor:
    """Haar-random orthonormal (d, rank) via QR of a Gaussian. fp32/CPU. A generic rank-r subspace."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, rank, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :rank]

def pooled_pca_subspace(acts_by_lang: dict[str, Tensor], rank: int) -> Tensor:
    """Top-`rank` principal directions of the POOLED (language-collapsed) activations:
    generic high-variance content directions, language-agnostic.

    Centered rows are L2-normalized before the SVD so this control is fit under the same
    angular, outlier-robust objective as `candidates.diffpca_subspace` (see `probe.unit_rows`);
    otherwise massive-activation dimensions dominate the pooled variance and distort the content
    floor. Kept comparable so the `pooled` vs language-subspace gap is a fair one.
    """
    X = torch.cat([a.to(LINALG_DEVICE, torch.float32) for a in acts_by_lang.values()], dim=0)
    Xc = X - X.mean(0, keepdim=True)
    Xn = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    _, _, Vh = torch.linalg.svd(Xn, full_matrices=False)
    return Vh[:rank].T

def permuted_label_subspace(acts_by_lang: dict[str, Tensor], rank: int, seed: int) -> Tensor:
    """Permutation null for 'M_s encodes language': shuffle which language each sample belongs to
    (preserving group sizes), recompute per-group means, build the two-step-SVD subspace."""
    langs = list(acts_by_lang)
    sizes = [acts_by_lang[l].shape[0] for l in langs]
    X = torch.cat([acts_by_lang[l].to(LINALG_DEVICE, torch.float32) for l in langs], dim=0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xs = X[torch.randperm(X.shape[0], generator=g)]
    chunks, off = {}, 0
    for l, n in zip(langs, sizes):
        chunks[l] = Xs[off:off + n]; off += n
    return build_language_subspace(language_means(chunks), rank)[1]
