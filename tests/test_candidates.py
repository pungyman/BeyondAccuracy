"""Unit tests for the five-subspace builder (no model, no data; synthetic activations).

The planted model is additive: h_i^l = c_i + g_l with g_l drawn inside a known r-dim
language-contrast subspace G. Then both `none` (centered means) and `diffpca` (within-problem
residuals) must recover G *exactly* -- content c_i cancels under centering / per-problem
differencing -- which is the geometric claim E3/E4 lean on.
"""
import numpy as np
import torch

from geometry import projector_distance, subspace_overlap
from candidates import (
    CANDIDATE_NAMES,
    build_candidate_subspaces,
    build_layer_subspaces,
    layer_slice,
    pooled_long_form,
)


def _planted(d=24, r=3, L=4, n=30, with_content=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    G, _ = torch.linalg.qr(torch.randn(d, r, generator=g))   # (d, r) orthonormal basis
    G = G[:, :r]
    gl = G @ torch.randn(r, L, generator=g)                  # (d, L): each g_l in span(G)
    # Content is SHARED across languages (parallel translations of the same problem); this is
    # what lets centering / per-problem differencing cancel it and expose G.
    content = torch.randn(n, d, generator=g) if with_content else torch.zeros(n, d)
    langs = [f"L{i}" for i in range(L)]
    acts = {lang: content + gl[:, li] for li, lang in enumerate(langs)}   # (n, d) each
    return acts, G, langs, n, L, r


def test_pooled_long_form_shapes_and_indices():
    acts, _, langs, n, L, _ = _planted()
    X, lang_idx, problem_idx = pooled_long_form(acts)
    assert X.shape == (n * L, next(iter(acts.values())).shape[1])
    assert torch.equal(lang_idx, torch.arange(L).repeat_interleave(n))
    assert torch.equal(problem_idx, torch.arange(n).repeat(L))
    # problem p's L parallel rows sit at p, n+p, 2n+p, ... (what residualize_problem groups)
    assert (problem_idx[[0, n, 2 * n]] == 0).all()


def test_build_candidate_shapes_and_orthonormality():
    acts, _, _, _, _, r = _planted()
    d = next(iter(acts.values())).shape[1]
    subs, meta = build_candidate_subspaces(acts, rank=r, seed=0)
    assert set(subs) == set(CANDIDATE_NAMES)
    for name, M in subs.items():
        assert M.shape == (d, r), name
        gram = M.T @ M
        assert torch.allclose(gram, torch.eye(r), atol=1e-4), f"{name} not orthonormal"
    scree = meta["diffpca_scree"]
    assert len(scree) == r and scree.sum() <= 1.0 + 1e-6


def test_none_and_diffpca_recover_planted_subspace_through_content():
    """With content present, both centered-means PCA and DiffPCA must still equal G."""
    acts, G, _, _, _, r = _planted(with_content=True)
    subs, _ = build_candidate_subspaces(acts, rank=r, seed=0)
    assert projector_distance(subs["none"], G) < 5e-3
    assert projector_distance(subs["diffpca"], G) < 5e-3
    assert projector_distance(subs["none"], subs["diffpca"]) < 5e-3
    assert subspace_overlap(subs["diffpca"], G) > 1 - 5e-3


def test_all_means_methods_agree_without_content():
    """No content offset -> alg1's anchor refinement has nothing to remove; all means-based
    constructions and DiffPCA collapse onto G."""
    acts, G, _, _, _, r = _planted(with_content=False)
    subs, _ = build_candidate_subspaces(acts, rank=r, seed=0)
    for name in ("alg1", "none", "diffpca"):
        assert projector_distance(subs[name], G) < 5e-3, name


def test_random_subspace_is_reproducible_and_generic():
    acts, G, _, _, _, r = _planted()
    s0, _ = build_candidate_subspaces(acts, rank=r, seed=7)
    s1, _ = build_candidate_subspaces(acts, rank=r, seed=7)
    assert torch.allclose(s0["random"], s1["random"])
    # a generic random subspace should not align with the planted one
    assert subspace_overlap(s0["random"], G) < 0.5


def _planted_layered(d=24, r=3, L=4, n=30, n_layers=5, seed=0):
    """Stack the single-layer planted model into a `{lang: (n, n_layers, d)}` cache; each layer k
    plants the SAME contrast subspace G rotated by a fixed per-layer generator, so every layer's
    recovered `M_s` should still equal single-layer construction on that slice."""
    g = torch.Generator().manual_seed(seed)
    acts = {f"L{i}": torch.zeros(n, n_layers, d) for i in range(L)}
    per_layer_G = []
    for k in range(n_layers):
        G, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
        G = G[:, :r]
        per_layer_G.append(G)
        gl = G @ torch.randn(r, L, generator=g)
        content = torch.randn(n, d, generator=g)
        for li in range(L):
            acts[f"L{li}"][:, k, :] = content + gl[:, li]
    return acts, per_layer_G, n_layers


def test_layer_slice_matches_manual_index():
    acts, _, n_layers = _planted_layered()
    sl = layer_slice(acts, 2)
    for lang, a in acts.items():
        assert torch.equal(sl[lang], a[:, 2, :])


def test_build_layer_subspaces_per_layer_orthonormal_and_matches_single_layer():
    acts, _, n_layers = _planted_layered()
    d = next(iter(acts.values())).shape[2]
    r = 3
    per_layer = build_layer_subspaces(acts, range(n_layers), name="alg1", rank=r)
    assert set(per_layer) == set(range(n_layers))
    for k, M in per_layer.items():
        assert M.shape == (d, r), k
        assert torch.allclose(M.T @ M, torch.eye(r), atol=1e-4), f"layer {k} not orthonormal"
        # identical to building the single candidate on that layer's slice directly
        ref, _ = build_candidate_subspaces(layer_slice(acts, k), rank=r, include=("alg1",))
        assert projector_distance(M, ref["alg1"]) < 5e-3   # equal up to principal-angle noise


def test_build_layer_subspaces_default_rank_is_L_minus_1():
    acts, _, n_layers = _planted_layered(L=4)
    per_layer = build_layer_subspaces(acts, [0], name="alg1")   # rank=None -> L-1
    assert per_layer[0].shape[1] == len(acts) - 1
