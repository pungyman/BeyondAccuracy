"""Unit tests for geometry helpers (pure linalg / stats; no model, no data)."""
import numpy as np
import torch

from geometry import (
    split_problems,
    ordered_or_unordered_pairs,
    captured_fraction,
    principal_angles,
    projector_distance,
    subspace_overlap,
    bootstrap_ci,
)

LANGS_11 = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def _orthonormal(d, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    return Q[:, :r]


def test_captured_fraction_in_and_out_of_subspace():
    d, r = 16, 4
    M = _orthonormal(d, r)
    coeffs = torch.randn(5, r)
    V_in = coeffs @ M.T                       # fully inside span(M)
    assert torch.allclose(captured_fraction(V_in, M), torch.ones(5), atol=1e-5)
    Mc = _orthonormal(d, d)                    # full basis; complement = columns r:
    V_out = torch.randn(5, d - r) @ Mc[:, r:].T
    # build M from the first r of the same basis so V_out is orthogonal to it
    assert captured_fraction(V_out, Mc[:, :r]).max() < 1e-5


def test_captured_fraction_random_floor():
    d, r = 256, 10
    M = _orthonormal(d, r, seed=1)
    V = torch.randn(2000, d, generator=torch.Generator().manual_seed(2))
    mean_rho = captured_fraction(V, M).mean().item()
    assert abs(mean_rho - r / d) < 0.01      # E[rho] ~ r/d for arbitrary vectors


def test_split_problems_disjoint_and_complete():
    tr, te = split_problems(150, frac=0.5, seed=0)
    assert len(tr) == 75 and len(te) == 75
    assert set(tr).isdisjoint(set(te))
    assert set(tr) | set(te) == set(range(150))
    # reproducible
    tr2, _ = split_problems(150, frac=0.5, seed=0)
    assert np.array_equal(tr, tr2)


def test_pair_counts():
    assert len(ordered_or_unordered_pairs(LANGS_11, ordered=False)) == 55
    assert len(ordered_or_unordered_pairs(LANGS_11, ordered=True)) == 110
    assert len(ordered_or_unordered_pairs(["a", "b", "c", "d"], ordered=False)) == 6
    assert len(ordered_or_unordered_pairs(["a", "b", "c", "d"], ordered=True)) == 12


def test_principal_angles_identical_subspace():
    M = _orthonormal(32, 6, seed=3)
    ang = principal_angles(M, M)
    # arccos(cos) loses ~sqrt(eps) precision in fp32 when cos ~ 1, so angles ~ 1e-3, not 0.
    assert torch.allclose(ang, torch.zeros(6), atol=2e-3)
    assert projector_distance(M, M) < 5e-3
    assert abs(subspace_overlap(M, M) - 1.0) < 1e-4
    # rotation within the span is invariant (fp32 arccos floor ~ 1e-3)
    R, _ = torch.linalg.qr(torch.randn(6, 6, generator=torch.Generator().manual_seed(4)))
    assert projector_distance(M, M @ R) < 5e-3


def test_principal_angles_orthogonal_subspaces():
    d, r = 32, 6
    B = _orthonormal(d, 2 * r, seed=5)
    Ma, Mb = B[:, :r], B[:, r:2 * r]          # orthogonal by construction
    ang = principal_angles(Ma, Mb)
    assert torch.allclose(ang, torch.full((r,), np.pi / 2), atol=1e-4)
    assert subspace_overlap(Ma, Mb) < 1e-4
    assert abs(projector_distance(Ma, Mb) - np.sqrt(2 * r)) < 1e-3


def test_bootstrap_ci_brackets_mean():
    x = np.linspace(0.0, 1.0, 200)
    mean, lo, hi = bootstrap_ci(x, n_boot=500, seed=0)
    assert abs(mean - 0.5) < 1e-6
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_cluster_widens_under_within_cluster_correlation():
    from geometry import bootstrap_ci, bootstrap_ci_cluster
    # 100 problems x 4 parallel "translations" that are perfectly correlated (identical rows) —
    # effective n is 100 problems, not 400 items. The iid CI (sees n=400) is ~2x too narrow.
    rng = np.random.default_rng(0)
    prob = rng.random(100)
    vals = np.repeat(prob, 4)
    clusters = np.repeat(np.arange(100), 4)
    m_i, lo_i, hi_i = bootstrap_ci(vals, n_boot=600, seed=1)
    m_c, lo_c, hi_c = bootstrap_ci_cluster(vals, clusters, n_boot=600, seed=1)
    assert abs(m_c - m_i) < 1e-9                       # same point estimate
    assert (hi_c - lo_c) > 1.5 * (hi_i - lo_i)         # clustered CI is materially wider


def test_bootstrap_ci_cluster_matches_iid_when_singletons():
    from geometry import bootstrap_ci, bootstrap_ci_cluster
    x = np.linspace(0, 1, 40)
    clusters = np.arange(40)                            # one item per cluster -> reduces to iid
    _, lo_i, hi_i = bootstrap_ci(x, n_boot=500, seed=3)
    _, lo_c, hi_c = bootstrap_ci_cluster(x, clusters, n_boot=500, seed=3)
    assert abs(lo_c - lo_i) < 0.05 and abs(hi_c - hi_i) < 0.05


def test_bootstrap_pvalue_cluster_degenerate_tails():
    import math
    import pytest
    from geometry import bootstrap_pvalue_cluster
    clusters = np.repeat(np.arange(50), 2)                       # 50 problems x 2 parallel rows
    pos = np.repeat(np.random.default_rng(0).random(50) + 0.1, 2)   # every value > 0
    neg = -pos                                                   # every value < 0
    # side='greater' (H1: mean > 0): all-positive -> p ~ 0; all-negative -> p ~ 1
    assert bootstrap_pvalue_cluster(pos, clusters, side="greater", n_boot=500, seed=1) < 0.01
    assert bootstrap_pvalue_cluster(neg, clusters, side="greater", n_boot=500, seed=1) > 0.99
    # side='less' mirrors it
    assert bootstrap_pvalue_cluster(neg, clusters, side="less", n_boot=500, seed=1) < 0.01
    # p is a valid tail probability, never exactly 0 (Davison–Hinkley +1)
    assert 0.0 < bootstrap_pvalue_cluster(pos, clusters, n_boot=500, seed=1) <= 1.0
    assert math.isnan(bootstrap_pvalue_cluster([], [], n_boot=10))
    with pytest.raises(ValueError):
        bootstrap_pvalue_cluster(pos, clusters, side="bogus", n_boot=10)
