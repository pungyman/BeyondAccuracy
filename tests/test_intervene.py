"""Unit tests for intervention helpers (pure linalg; no model, no data)."""
import torch

from intervene import (
    InterventionSpec,
    MaskHolder,
    ablate,
    build_layer_hooks,
    edit_mask,
)


def _orthonormal(d, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    return Q[:, :r]


def test_ablate_lam_zero_is_identity():
    d, r = 16, 4
    M = _orthonormal(d, r)
    h = torch.randn(5, d, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(ablate(h, M, lam=0.0), h, atol=1e-6)


def test_ablate_lam_one_matches_full_removal():
    d, r = 16, 4
    M = _orthonormal(d, r)
    h = torch.randn(5, d, generator=torch.Generator().manual_seed(2))
    expected = h - (h @ M) @ M.T
    assert torch.allclose(ablate(h, M, lam=1.0), expected, atol=1e-6)
    # default lam is 1.0
    assert torch.allclose(ablate(h, M), expected, atol=1e-6)


def test_ablate_lam_one_removes_subspace_component():
    d, r = 16, 4
    M = _orthonormal(d, r)
    h = torch.randn(5, d, generator=torch.Generator().manual_seed(3))
    out = ablate(h, M, lam=1.0)
    # residual should have no component in span(M)
    assert torch.allclose(out @ M, torch.zeros(5, r), atol=1e-5)


def test_ablate_lam_scales_linearly():
    d, r = 16, 4
    M = _orthonormal(d, r)
    h = torch.randn(5, d, generator=torch.Generator().manual_seed(4))
    comp = (h @ M) @ M.T
    for lam in (0.25, 0.5, 0.75, 1.25):
        assert torch.allclose(ablate(h, M, lam=lam), h - lam * comp, atol=1e-6)


def test_ablate_negative_lam_reinjects():
    """lam < 0 amplifies the subspace component (the high-layer re-injection band)."""
    d, r = 16, 4
    M = _orthonormal(d, r)
    h = torch.randn(5, d, generator=torch.Generator().manual_seed(5))
    comp = (h @ M) @ M.T
    out = ablate(h, M, lam=-0.3)
    # projection of the output onto span(M) is 1.3x the input's (grew, not shrank)
    assert torch.allclose((out @ M), 1.3 * (h @ M), atol=1e-5)
    assert torch.allclose(out, h + 0.3 * comp, atol=1e-6)


def _two_band_spec(d=16, r=4, n_layers=8):
    subs = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    # mid band [2,4) ablate (+), high band [4,6) reinject (-), rest untouched (0)
    lam = {2: 0.3, 3: 0.3, 4: -0.2, 5: -0.2, 6: 0.0}
    return InterventionSpec(subspaces=subs, lam=lam), subs


def test_build_layer_hooks_lam_signs_and_dropped_zeros():
    spec, _ = _two_band_spec()
    hooks = build_layer_hooks(spec)
    # zero-λ layer is dropped; the two bands survive with the right signs
    assert set(hooks) == {2, 3, 4, 5}
    d = spec.subspaces[2].shape[0]
    h = torch.randn(1, 3, d, generator=torch.Generator().manual_seed(9))  # prefill (seq>1)
    for layer in (2, 3):                                                   # mid: ablate shrinks
        M = spec.subspaces[layer]
        assert (hooks[layer](h) @ M).abs().sum() < (h @ M).abs().sum()
    for layer in (4, 5):                                                   # high: reinject grows
        M = spec.subspaces[layer]
        assert (hooks[layer](h) @ M).abs().sum() > (h @ M).abs().sum()


def test_prompt_only_applies_on_prefill_but_not_decode():
    spec, _ = _two_band_spec()
    assert spec.scope == "prompt_only"
    fn = build_layer_hooks(spec)[2]
    d = spec.subspaces[2].shape[0]
    prefill = torch.randn(1, 5, d, generator=torch.Generator().manual_seed(10))  # seq>1
    decode = torch.randn(1, 1, d, generator=torch.Generator().manual_seed(11))   # seq==1
    assert not torch.allclose(fn(prefill), prefill)      # prefill edited
    assert torch.allclose(fn(decode), decode)            # decode step untouched


def test_all_steps_applies_on_decode_too():
    subs = {2: _orthonormal(16, 4, seed=2)}
    spec = InterventionSpec(subspaces=subs, lam={2: 0.3}, scope="all_steps")
    fn = build_layer_hooks(spec)[2]
    decode = torch.randn(1, 1, 16, generator=torch.Generator().manual_seed(12))
    assert not torch.allclose(fn(decode), decode)        # all_steps edits the decode step


def test_edit_mask_context_only_vs_all():
    m = edit_mask(seq_len=6, ctx_len=4, scope="prompt_only")
    assert m.tolist() == [True, True, True, True, False, False]
    assert edit_mask(6, 4, scope="all_steps").all()


def test_mask_holder_edits_only_context_positions():
    """A prompt-only edit_mask via MaskHolder leaves the candidate positions bit-identical."""
    d, r = 16, 4
    subs = {0: _orthonormal(d, r, seed=0)}
    spec = InterventionSpec(subspaces=subs, lam={0: 0.5}, scope="prompt_only")
    mh = MaskHolder(scope="prompt_only")
    fn = build_layer_hooks(spec, mask_holder=mh)[0]
    seq, ctx = 6, 4
    mh.set_from_ctx(seq, ctx)
    h = torch.randn(1, seq, d, generator=torch.Generator().manual_seed(13))
    out = fn(h)
    assert not torch.allclose(out[:, :ctx], h[:, :ctx])  # context positions edited
    assert torch.allclose(out[:, ctx:], h[:, ctx:])      # candidate positions untouched
    # cleared holder -> generative branch: a seq>1 tensor is treated as prefill and edited fully
    mh.clear()
    out2 = fn(h)
    assert not torch.allclose(out2[:, ctx:], h[:, ctx:])
