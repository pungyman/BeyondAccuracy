"""Unit tests for the E6 affine-edit estimators + affine hook path (synthetic data only).

Planted models mirror test_candidates.py: additive h_i^l = c_i + g_l (+ noise), with the
language offsets g_l inside a known contrast span. The load-bearing claims tested here:

- LEACE *guards*: at lam=1 the post-edit language means coincide (no linear probe beats chance)
  — and it does so with an edit no larger (in mean squared norm) than the orthogonal projection
  of the same span, strictly smaller under anisotropic nuisance covariance.
- Wiener targets only the language component: (near-)zero gain off the language span, strong
  mean-separation shrinkage at high SNR.
- Transports match their target moments at lam=1 (mean / per-dim std / full covariance).
- The paired translator actually predicts the paired counterfactual (in-sample error way below
  the untranslated distance) and interpolates to identity at lam=0.
- AffineSpec/build_affine_hooks reproduce the InterventionSpec scope/mask semantics exactly.
"""
import numpy as np
import pytest
import torch

from intervene import AffineSpec, MaskHolder, ablate, build_affine_hooks
from operators import (
    EDIT_NAMES,
    FAMILY_BASES,
    FAMILY_OPERATORS,
    TRANSPORT_NAMES,
    adain_edit,
    build_family_edits,
    build_layer_edits,
    family_cell_absent,
    layer_whitener,
    edit_energy,
    gaussian_ot_edit,
    leace_edit,
    meanshift_edit,
    projection_edit,
    translator_edit,
    uniform_shrink_edit,
    wdiffpca_edit,
    wiener_edit,
)


def _planted(d=24, r=3, L=4, n=60, noise=0.0, aniso=None, lang_scale=1.0, inter=None, seed=0):
    """Additive planted model; `aniso` adds a high-variance nuisance direction to the content so
    the Euclidean and whitened geometries disagree; `lang_scale` sets the language-offset size
    relative to the unit-variance content (the language SNR); `inter=(r_int, scale)` adds a
    structured content x language interaction in its own span (returned third)."""
    g = torch.Generator().manual_seed(seed)
    G, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    G = G[:, :r]
    gl = lang_scale * (G @ torch.randn(r, L, generator=g))
    content = torch.randn(n, d, generator=g)
    if aniso is not None:
        u, scale = torch.randn(d, generator=g), aniso
        u = u / u.norm()
        content = content + scale * torch.randn(n, 1, generator=g) * u
    Q = None
    if inter is not None:
        r_int, _ = inter
        Q, _ = torch.linalg.qr(torch.randn(d, r_int, generator=g))
        Q = Q[:, :r_int]
    langs = [f"L{i}" for i in range(L)]
    acts = {}
    for li, lang in enumerate(langs):
        eps = noise * torch.randn(n, d, generator=g)
        row = content + gl[:, li] + eps
        if inter is not None:
            r_int, scale = inter
            row = row + scale * torch.randn(n, r_int, generator=g) @ Q.T
        acts[lang] = row
    _planted.last_inter_span = Q     # interaction span for tests that need it
    return acts, G, langs


def _pooled(acts):
    return torch.cat(list(acts.values()), dim=0)


def _lang_mean_spread(X_by_lang):
    """Max distance of a per-language mean from the pooled mean (0 = perfectly guarded)."""
    means = torch.stack([a.mean(0) for a in X_by_lang.values()])
    return (means - means.mean(0)).norm(dim=1).max().item()


def _apply_all(edit, acts, lam=1.0):
    return {lang: edit.apply(a, lam=lam) for lang, a in acts.items()}


# --- AffineEdit basics ----------------------------------------------------------------------------


def test_projection_edit_matches_ablate_exactly():
    acts, G, _ = _planted()
    h = acts["L0"]
    for lam in (0.0, 0.3, 1.0, -0.2):
        assert torch.allclose(projection_edit(G).apply(h, lam=lam), ablate(h, G, lam=lam),
                              atol=1e-6)


def test_affine_edit_lam_zero_is_identity_and_scales_linearly():
    acts, _, _ = _planted()
    edit = leace_edit(acts, shrink=1e-6)
    h = acts["L1"]
    assert torch.allclose(edit.apply(h, lam=0.0), h, atol=1e-6)
    delta = edit.delta(h)
    for lam in (0.25, 1.0, -0.5):
        assert torch.allclose(edit.apply(h, lam=lam), h - lam * delta, atol=1e-5)


# --- LEACE ---------------------------------------------------------------------------------------


def test_leace_guards_class_means():
    acts, _, _ = _planted(noise=0.05)
    edit = leace_edit(acts, shrink=1e-6)
    spread_before = _lang_mean_spread(acts)
    spread_after = _lang_mean_spread(_apply_all(edit, acts))
    assert spread_after < 1e-3 * spread_before


def test_leace_edit_energy_below_orthogonal_projection_under_anisotropy():
    """Same guarding target, smaller edit: the whitened-oblique projection must move the data
    (mean-squared) no more than hard orthogonal removal of the mean span — strictly less when a
    high-variance nuisance direction overlaps the span."""
    acts, _, _ = _planted(noise=0.05, aniso=8.0, seed=3)
    X = _pooled(acts)
    # orthogonal removal of the centered-mean span (the `none`/alg1-style operator)
    means = torch.stack([a.mean(0) for a in acts.values()])
    Delta = (means - means.mean(0)).T
    Q, _ = torch.linalg.qr(Delta)
    Q = Q[:, : means.shape[0] - 1]
    mu = X.mean(0)
    orth_energy = ((((X - mu) @ Q) @ Q.T) ** 2).sum()
    # both remove the mean spread; orthogonal projection also guards here, so LEACE (the
    # minimal-MSE guard) must not exceed its edit energy
    guarded = _apply_all(projection_edit(Q), {l: a - mu for l, a in acts.items()})
    assert _lang_mean_spread(guarded) < 1e-4
    edit = leace_edit(acts, shrink=1e-6)
    leace_energy = ((X - edit.apply(X)) ** 2).sum()
    assert leace_energy < orth_energy * 1.0001
    assert leace_energy < 0.9 * orth_energy  # strictly smaller under the planted anisotropy


# --- Wiener --------------------------------------------------------------------------------------


def test_wiener_gain_targets_the_language_span():
    """The gain must attenuate along the planted language span far more than off it (exact
    zero off-span only holds at infinite n — sample-covariance noise leaks a little)."""
    acts, G, _ = _planted(noise=0.05, n=200, seed=1)
    edit = wiener_edit(acts, shrink=1e-6)
    d = G.shape[0]
    g = torch.Generator().manual_seed(9)
    v = torch.randn(1, d, generator=g)
    v_off = v - (v @ G) @ G.T
    v_on = (v @ G) @ G.T
    mu = _pooled(acts).mean(0)
    gain_off = edit.delta(v_off + mu).norm() / v_off.norm()
    gain_on = edit.delta(v_on + mu).norm() / v_on.norm()
    assert gain_on > 0.2
    assert gain_off < 0.5 * gain_on


def test_wiener_shrinkage_tracks_language_snr():
    """Attenuation is snr/(1+snr): near-total at high language SNR, partial (never full)
    at snr ~ 1 — the qualitative difference from the hard projection."""
    high, _, _ = _planted(noise=0.02, n=120, lang_scale=8.0, seed=2)
    edit = wiener_edit(high, shrink=1e-6)
    assert _lang_mean_spread(_apply_all(edit, high)) < 0.1 * _lang_mean_spread(high)
    low, _, _ = _planted(noise=0.02, n=120, lang_scale=1.0, seed=2)
    edit_low = wiener_edit(low, shrink=1e-6)
    ratio = _lang_mean_spread(_apply_all(edit_low, low)) / _lang_mean_spread(low)
    assert 0.2 < ratio < 0.8


# --- transports ----------------------------------------------------------------------------------


def test_meanshift_moves_source_mean_to_target():
    acts, _, _ = _planted(noise=0.05)
    edit = meanshift_edit(acts, source="L0", target="L2")
    moved = edit.apply(acts["L0"])
    assert (moved.mean(0) - acts["L2"].mean(0)).norm() < 1e-4
    # per-row shape is preserved exactly (a pure translation)
    assert torch.allclose(moved - moved.mean(0), acts["L0"] - acts["L0"].mean(0), atol=1e-5)


def test_adain_matches_mean_and_std():
    acts, _, _ = _planted(noise=0.3, seed=4)
    # give the source a distinct per-dim scale so the diagonal term is exercised
    acts["L0"] = acts["L0"] * torch.linspace(0.5, 2.0, acts["L0"].shape[1])
    edit = adain_edit(acts, source="L0", target="L1")
    moved = edit.apply(acts["L0"])
    assert (moved.mean(0) - acts["L1"].mean(0)).norm() < 1e-4
    assert torch.allclose(moved.std(0), acts["L1"].std(0), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("kind", ["bures", "coral"])
def test_gaussian_ot_matches_full_covariance(kind):
    acts, _, _ = _planted(d=12, n=400, noise=0.5, aniso=3.0, seed=5)
    edit = gaussian_ot_edit(acts, source="L0", target="L1", kind=kind, shrink=1e-8)
    moved = edit.apply(acts["L0"]).double()
    cov_moved = torch.cov(moved.T)
    cov_target = torch.cov(acts["L1"].double().T)
    assert (moved.mean(0).float() - acts["L1"].mean(0)).norm() < 1e-3
    assert (cov_moved - cov_target).norm() < 0.05 * cov_target.norm()


def test_gaussian_ot_rank_truncation_approximates_dense():
    acts, _, _ = _planted(d=12, n=400, noise=0.5, seed=6)
    dense = gaussian_ot_edit(acts, source="L0", target="L1", shrink=1e-6)
    trunc = gaussian_ot_edit(acts, source="L0", target="L1", shrink=1e-6, rank=12)  # full rank
    h = acts["L0"][:5]
    assert torch.allclose(dense.apply(h), trunc.apply(h), atol=1e-4)


def test_translator_predicts_paired_counterfactual_in_sample():
    acts, _, _ = _planted(noise=0.05, n=80, seed=7)
    edit = translator_edit(acts, source="L0", target="L2", ridge=1e-3)
    moved = edit.apply(acts["L0"])                    # lam=1 -> T(h)
    err = (moved - acts["L2"]).norm()
    base = (acts["L0"] - acts["L2"]).norm()
    assert err < 0.35 * base                          # paired fit beats the untranslated gap
    assert torch.allclose(edit.apply(acts["L0"], lam=0.0), acts["L0"], atol=1e-6)
    # rank-truncated correction stays close to the full solve
    trunc = translator_edit(acts, source="L0", target="L2", ridge=1e-3, rank=40)
    assert (trunc.apply(acts["L0"]) - moved).norm() < 0.2 * err + 1e-4


# --- edit_energy (the dose unit) -------------------------------------------------------------------


def test_edit_energy_matches_analytic():
    acts, G, _ = _planted(noise=0.1)
    X = _pooled(acts)
    # hard projection: energy = captured fraction of the rows
    direct = float(((X @ G) ** 2).sum() / (X ** 2).sum())
    assert abs(edit_energy(projection_edit(G), X) - direct) < 1e-6
    # uniform shrink by eps: energy = eps^2 exactly
    from intervene import AffineEdit

    shrink = AffineEdit(diag=torch.full((X.shape[1],), 0.3))
    assert abs(edit_energy(shrink, X) - 0.09) < 1e-6


# --- matrix-completion cells: instance target under the better operators --------------------------


def test_wdiffpca_is_oblique_idempotent_and_ranked():
    acts, _, _ = _planted(noise=0.3, inter=(4, 2.0), seed=11)
    edit = wdiffpca_edit(acts, rank=3, shrink=1e-4)
    assert edit.U.shape[1] == 3 and edit.V.shape[1] == 3
    X = _pooled(acts)
    once = edit.apply(X)
    assert torch.allclose(edit.apply(once), once, atol=1e-3)   # E^2 = E (oblique projector)


def test_wdiffpca_targets_interaction_over_means():
    """With a strong structured interaction, the instance-target cell removes within-problem
    residual energy better than leace (mean-span target), and vice versa on mean spread —
    i.e. the two cells genuinely differ by TARGET, not by operator."""
    acts, _, _ = _planted(noise=0.2, lang_scale=0.7, inter=(4, 3.0), n=120, seed=12)
    wd = wdiffpca_edit(acts, rank=4, shrink=1e-4)
    le = leace_edit(acts, rank=3, shrink=1e-4)

    from candidates import pooled_long_form
    from probe import residualize_problem

    X_t, _, prob_t = pooled_long_form(acts)
    X = X_t.numpy()

    def resid_energy(edit):
        Xe = edit.apply(torch.tensor(X, dtype=torch.float32)).numpy()
        return (residualize_problem(Xe, prob_t.numpy()) ** 2).sum()

    assert resid_energy(wd) < resid_energy(le)                       # instance cell kills residuals
    assert (_lang_mean_spread(_apply_all(le, acts))
            < _lang_mean_spread(_apply_all(wd, acts)))               # mean cell kills mean spread


def test_wiener_instance_variant_runs_and_differs():
    acts, _, _ = _planted(noise=0.2, inter=(4, 3.0), n=120, seed=13)
    w_means = wiener_edit(acts, shrink=1e-4)
    w_inst = wiener_edit(acts, shrink=1e-4, sigma_g="instance")
    X = _pooled(acts)[:8]
    assert not torch.allclose(w_means.delta(X), w_inst.delta(X), atol=1e-3)
    with pytest.raises(ValueError):
        wiener_edit(acts, sigma_g="nope")


# --- uniform-shrink control ----------------------------------------------------------------------


def test_uniform_shrink_matches_alg1_edit_energy():
    acts, _, _ = _planted(noise=0.1, seed=8)
    edit = uniform_shrink_edit(acts)
    X = _pooled(acts)
    shrink_energy = ((X - edit.apply(X)) ** 2).sum()
    from subspace import build_language_subspace, language_means

    M = build_language_subspace(language_means(acts), len(acts) - 1)[1]
    alg1_energy = (((X @ M) @ M.T) ** 2).sum()
    assert abs(shrink_energy / alg1_energy - 1.0) < 1e-3


# --- per-layer builder ----------------------------------------------------------------------------


def _layered(acts, n_layers=3):
    return {l: a.unsqueeze(1).repeat(1, n_layers, 1) for l, a in acts.items()}


def test_build_layer_edits_language_agnostic_and_transport():
    acts, _, _ = _planted(noise=0.05)
    cache = _layered(acts)
    edits = build_layer_edits(cache, [0, 2], name="leace", shrink=1e-4)
    assert set(edits) == {0, 2}
    tr = build_layer_edits(cache, [1], name="meanshift", source="L0", target="barycenter")
    assert set(tr) == {1}
    with pytest.raises(ValueError):
        build_layer_edits(cache, [0], name="meanshift")          # transport without source
    with pytest.raises(ValueError):
        build_layer_edits(cache, [0], name="nope")
    assert set(EDIT_NAMES).isdisjoint(TRANSPORT_NAMES)


# --- affine hooks: scope/mask parity with InterventionSpec ----------------------------------------


def _bias_edit(d, val=1.0):
    from intervene import AffineEdit

    return AffineEdit(b=torch.full((d,), val))


def test_affine_hooks_drop_zero_lam_and_apply_signed_bands():
    d = 8
    spec = AffineSpec(edits={2: _bias_edit(d), 3: _bias_edit(d), 4: _bias_edit(d)},
                      lam={2: 0.5, 3: 0.0, 4: -0.5})
    hooks = build_affine_hooks(spec)
    assert set(hooks) == {2, 4}
    h = torch.zeros(1, 3, d)
    assert torch.allclose(hooks[2](h), torch.full_like(h, -0.5))   # remove the bias
    assert torch.allclose(hooks[4](h), torch.full_like(h, +0.5))   # negative lam re-injects


def test_affine_prompt_only_skips_decode_steps():
    d = 8
    spec = AffineSpec(edits={0: _bias_edit(d)}, lam={0: 1.0}, scope="prompt_only")
    fn = build_affine_hooks(spec)[0]
    prefill = torch.zeros(1, 4, d)
    decode = torch.zeros(1, 1, d)
    assert not torch.allclose(fn(prefill), prefill)
    assert torch.allclose(fn(decode), decode)
    all_steps = AffineSpec(edits={0: _bias_edit(d)}, lam={0: 1.0}, scope="all_steps")
    assert not torch.allclose(build_affine_hooks(all_steps)[0](decode), decode)


def test_coordinate_search_accepts_affine_edits():
    """The tuning searches must ride unchanged on `{layer -> AffineEdit}` mappings, freezing an
    `AffineSpec` (the E6 seam)."""
    import pandas as pd
    from tuning import coordinate_search

    d, n_layers = 8, 28
    edits = {l: _bias_edit(d, val=1.0) for l in range(n_layers)}

    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None,
             model_name=None, **kw):
        # recover the λ each hook applies to a zero prefill row (delta = b -> out = -λ·b)
        h = torch.zeros(1, 2, d)
        lams = [-float(fn(h)[0, 0, 0]) for fn in hooks.values()]
        lam_mid = max([v for v in lams if v > 0], default=0.0)
        acc = 1.0 - abs(lam_mid - 0.3)                       # peak at λ_mid = 0.3
        return pd.DataFrame([{"task": "mgsm", "lang": "en", "correct": acc}])

    spec, log = coordinate_search(
        model=None, tok=None, dev_items=[object()], subspaces=edits, n_layers=n_layers,
        model_tag="Qwen2.5-7B-Instruct", eval_fn=stub,
    )
    assert isinstance(spec, AffineSpec)
    assert {v for v in spec.lam.values() if v > 0} == {0.3}
    assert set(log["step"]) == {"lam_mid", "lam_high"}


def test_affine_mask_holder_edits_only_context_positions():
    d = 8
    spec = AffineSpec(edits={0: _bias_edit(d)}, lam={0: 1.0}, scope="prompt_only")
    mh = MaskHolder(scope="prompt_only")
    fn = build_affine_hooks(spec, mask_holder=mh)[0]
    seq, ctx = 6, 4
    mh.set_from_ctx(seq, ctx)
    h = torch.zeros(1, seq, d)
    out = fn(h)
    assert not torch.allclose(out[:, :ctx], h[:, :ctx])
    assert torch.allclose(out[:, ctx:], h[:, ctx:])


# --- diffPCA x operator cross (basis injection) --------------------------------------------------


def test_leace_basis_injection_uses_diffpca_span():
    from candidates import diffpca_subspace
    acts, _, _ = _planted(noise=0.05, aniso=8.0)             # anisotropy => whitened != Euclidean
    Ms, _ = diffpca_subspace(acts, rank=3, angular=True)
    native = leace_edit(acts)                                # mean-span oblique
    crossed = leace_edit(acts, basis=Ms)                     # diffpca-span oblique (the cross cell)
    h = acts["L0"]
    assert torch.isfinite(crossed.delta(h)).all()
    assert np.isfinite(edit_energy(crossed, _pooled(acts)))
    assert crossed.U.shape[1] == Ms.shape[1]                 # rank matched to the injected basis
    # injecting the diffPCA span changes the edit vs the native language-mean span
    assert not torch.allclose(crossed.delta(h), native.delta(h), atol=1e-4)


def test_wiener_basis_injection_valid_and_distinct():
    from candidates import diffpca_subspace
    acts, _, _ = _planted(noise=0.1, lang_scale=2.0)
    Ms, _ = diffpca_subspace(acts, rank=3, angular=True)
    native = wiener_edit(acts)                               # mean-target soft shrinkage
    crossed = wiener_edit(acts, basis=Ms)                    # diffpca-span soft shrinkage (cross cell)
    h = acts["L1"]
    assert torch.isfinite(crossed.delta(h)).all()
    assert np.isfinite(edit_energy(crossed, _pooled(acts)))
    assert crossed.U.shape[1] == Ms.shape[1]
    assert not torch.allclose(crossed.delta(h), native.delta(h), atol=1e-4)


def test_build_layer_edits_injects_basis_per_layer():
    from candidates import build_layer_subspaces
    torch.manual_seed(0)
    n, nl, d, L = 40, 3, 20, 4
    acts = {f"L{i}": torch.randn(n, nl, d) + (i + 1) * 0.2 for i in range(L)}   # (n, n_layers, d)
    ms = build_layer_subspaces(acts, range(nl), name="diffpca", rank=3)         # {layer -> (d, 3)}
    edits = build_layer_edits(acts, range(nl), name="leace", basis_by_layer=ms)
    assert set(edits) == {0, 1, 2}
    # each layer's built edit equals a direct leace_edit on that layer's slice with that basis
    sl0 = {lg: a[:, 0, :] for lg, a in acts.items()}
    direct = leace_edit(sl0, basis=ms[0])
    assert torch.allclose(edits[0].U, direct.U, atol=1e-5)


# --- E9: the (basis x operator) family join --------------------------------------------------------


def test_build_family_edits_covers_all_six_families():
    acts, _, _ = _planted(noise=0.05)
    cache = _layered(acts)
    cap = len(acts) - 1
    for basis in FAMILY_BASES:
        for operator in FAMILY_OPERATORS:
            edits = build_family_edits(cache, [0, 2], basis=basis, operator=operator, rank=cap)
            assert set(edits) == {0, 2}
            for e in edits.values():
                assert e.U is not None and e.V is not None
                assert e.U.shape[0] == acts["L0"].shape[1]          # (d, r)


def test_build_family_edits_rank_axis_bites_where_it_exists():
    acts, _, _ = _planted(noise=0.05)
    cache = _layered(acts)
    cap = len(acts) - 1
    for basis, operator in [("alg1", "proj"), ("alg1", "leace"),
                            ("diffpca", "proj"), ("diffpca", "leace"), ("diffpca", "wiener")]:
        lo = build_family_edits(cache, [0], basis=basis, operator=operator, rank=1)[0]
        hi = build_family_edits(cache, [0], basis=basis, operator=operator, rank=cap)[0]
        assert lo.U.shape[1] == 1 and hi.U.shape[1] == cap
        assert not torch.equal(lo.U, hi.U)


def test_alg1_wiener_has_no_rank_knob_and_is_refused_off_cap():
    """The finding that took the E9 grid from 39 arms to 37: wiener(means) ignores `rank`."""
    acts, _, _ = _planted(noise=0.05)
    cache = _layered(acts)
    cap = len(acts) - 1
    # the underlying estimator silently returns the SAME edit at every rank ...
    a = wiener_edit(acts, rank=1)
    b = wiener_edit(acts, rank=cap)
    assert torch.equal(a.U, b.U) and torch.equal(a.V, b.V)
    assert a.U.shape[1] == len(acts)                        # F = Delta/sqrt(L) is (d, L), not (d, rank)
    # ... so the family join refuses the cell rather than answering with a duplicate
    assert family_cell_absent("alg1", "wiener", cap, len(acts)) is None
    for r in (1, 2):
        assert family_cell_absent("alg1", "wiener", r, len(acts)) is not None
        with pytest.raises(ValueError, match="no rank knob"):
            build_family_edits(cache, [0], basis="alg1", operator="wiener", rank=r)


def test_alg1_is_capped_at_l_minus_one():
    acts, _, _ = _planted(noise=0.05)
    cache = _layered(acts)
    cap = len(acts) - 1
    for operator in FAMILY_OPERATORS:
        assert family_cell_absent("alg1", operator, cap + 1, len(acts)) is not None
        with pytest.raises(ValueError, match="at most"):
            build_family_edits(cache, [0], basis="alg1", operator=operator, rank=cap + 1)
    # diffpca has no such cap -- it carries the extra rank cell
    assert family_cell_absent("diffpca", "proj", cap + 5, len(acts)) is None


def test_family_cell_absent_rejects_unknown_names_and_bad_rank():
    assert family_cell_absent("nope", "proj", 3, 4) is not None
    assert family_cell_absent("alg1", "nope", 3, 4) is not None
    assert family_cell_absent("alg1", "proj", 0, 4) is not None


# --- whitener reuse: byte-identical to recomputing -------------------------------------------------


def test_layer_whitener_reuse_is_byte_identical():
    acts, _, _ = _planted(noise=0.05, aniso=3.0)
    d = acts["L0"].shape[1]
    W = layer_whitener(acts, shrink=0.1)
    B = torch.linalg.qr(torch.randn(d, 2, generator=torch.Generator().manual_seed(1)))[0][:, :2]
    cases = [
        lambda w: leace_edit(acts, whiten=w),
        lambda w: leace_edit(acts, rank=2, whiten=w),
        lambda w: leace_edit(acts, basis=B, whiten=w),
        lambda w: wiener_edit(acts, whiten=w),
        lambda w: wiener_edit(acts, basis=B, whiten=w),
        lambda w: wiener_edit(acts, sigma_g="instance", rank=2, whiten=w),
    ]
    for fn in cases:
        fresh, cached = fn(None), fn(W)
        assert torch.equal(fresh.c, cached.c)
        assert torch.equal(fresh.U, cached.U)
        assert torch.equal(fresh.V, cached.V)


def test_layer_whitener_fails_loud_on_shrink_mismatch_and_missing_block():
    acts, _, _ = _planted(noise=0.05)
    W = layer_whitener(acts, shrink=0.1)
    with pytest.raises(ValueError, match="shrink"):
        leace_edit(acts, shrink=0.01, whiten=W)             # ridge is baked into W_inv
    thin = layer_whitener(acts, shrink=0.1, need=("leace",))
    leace_edit(acts, whiten=thin)                           # has what it needs
    with pytest.raises(ValueError, match="lacks"):
        wiener_edit(acts, whiten=thin)                       # no S_inv_means in a leace-only whitener


def test_build_layer_edits_forwards_whitener_per_layer():
    acts, _, _ = _planted(noise=0.05)
    cache = _layered(acts)
    W = {l: layer_whitener({k: v[:, l, :] for k, v in cache.items()}, shrink=0.1) for l in (0, 2)}
    plain = build_layer_edits(cache, [0, 2], name="leace")
    reused = build_layer_edits(cache, [0, 2], name="leace", whitener_by_layer=W)
    for l in (0, 2):
        assert torch.equal(plain[l].U, reused[l].U)
        assert torch.equal(plain[l].V, reused[l].V)
