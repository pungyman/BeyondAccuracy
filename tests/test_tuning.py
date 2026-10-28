"""Unit tests for the lever-2 λ-schedule search (pure/stubbed; no model, no data)."""
import pandas as pd
import torch

import numpy as np

from tuning import (
    LAM_SWEEP_SYMMETRIC,
    coordinate_search,
    grid_search,
    lam_from_params,
    layer_sweep,
    make_bands,
    make_schedule,
    optuna_search,
    optuna_warmstart_delta,
    paper_fixed_schedule,
    pareto_front,
    place_bands,
    rank_sweep,
    rank_sweep_retuned,
    select_rank_1se,
    strength_sweep,
    trough_layer,
)


def _orthonormal(d, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    return Q[:, :r]


# --- make_schedule -----------------------------------------------------------------------------

def test_make_schedule_signs_and_coverage():
    sched = make_schedule(mid_start=4, mid_end=8, lam_mid=0.3,
                          high_end=12, lam_high=-0.2, n_layers=16)
    assert set(range(4, 8)) <= set(sched) and all(sched[l] == 0.3 for l in range(4, 8))
    assert set(range(8, 12)) <= set(sched) and all(sched[l] == -0.2 for l in range(8, 12))
    assert all(l not in sched for l in range(0, 4))       # below the mid band: untouched
    assert all(l not in sched for l in range(12, 16))     # above the high band: untouched


def test_make_schedule_ablate_only_drops_high_band():
    sched = make_schedule(4, 8, 0.3, 12, 0.0, n_layers=16)  # lam_high=0
    assert set(sched) == set(range(4, 8))                   # only the mid band survives


def test_make_schedule_clamps_to_layer_range():
    sched = make_schedule(-3, 4, 0.5, 40, -0.1, n_layers=8)
    assert min(sched) >= 0 and max(sched) < 8


# --- place_bands / trough_layer ----------------------------------------------------------------

def test_place_bands_centered_on_trough_and_in_range():
    n_layers = 28
    trough = 18
    mid_start, mid_end, high_end = place_bands(trough, n_layers)
    assert 0 <= mid_start < trough < mid_end <= high_end == n_layers
    # the trough sits inside the ablate band
    assert mid_start <= trough < mid_end


def test_place_bands_half_width_override():
    mid_start, mid_end, high_end = place_bands(10, 20, half_width=3)
    assert (mid_start, mid_end, high_end) == (7, 13, 20)


def test_trough_layer_known_and_fallback():
    assert trough_layer("Qwen2.5-1.5B-Instruct", 28) == 16
    assert trough_layer("Unlisted-Model", 30) == round(0.6 * 30)


# --- paper_fixed_schedule ----------------------------------------------------------------------

def test_paper_fixed_schedule_two_signed_bands():
    sched = paper_fixed_schedule(28)
    assert any(v > 0 for v in sched.values()) and any(v < 0 for v in sched.values())
    assert all(0 <= l < 28 for l in sched)
    # on the reference depth the bands land on the paper's layers
    assert sched[15] == 0.3 and sched[25] == -0.3


# --- coordinate_search (stubbed run_eval; toy scorer) ------------------------------------------

def _recover_lam(hook_fn, M):
    """Recover the λ a hook applies: ablate((1-... )) on the first basis vector row of M."""
    d = M.shape[0]
    h = M[:, 0].view(1, 1, d).expand(1, 2, d).contiguous()   # seq=2 -> prefill branch
    out = hook_fn(h)
    return 1.0 - float(out[0, 0] @ M[:, 0])                   # out row = (1-λ)·M[:,0]


def _toy_eval_factory(subspaces, target_mid, target_high):
    """A stub `run_eval`: accuracy is a tent peaked at (target_mid, target_high), read back from
    the λ the hooks actually apply — so the search's argmax is fully determined and checkable."""
    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None,
             model_name=None, **kw):
        lams = {l: _recover_lam(fn, subspaces[l]) for l, fn in hooks.items()}
        pos = [v for v in lams.values() if v > 1e-9]
        neg = [v for v in lams.values() if v < -1e-9]
        lam_mid = max(pos) if pos else 0.0
        lam_high = min(neg) if neg else 0.0
        acc = 1.0 - 0.5 * abs(lam_mid - target_mid) - 0.5 * abs(lam_high - target_high)
        return pd.DataFrame([{"model": "stub", "task": "mgsm", "lang": "en", "correct": acc}])
    return stub


def test_coordinate_search_picks_argmax_on_toy_scorer():
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    target_mid, target_high = 0.4, -0.2            # both inside the default grids
    stub = _toy_eval_factory(subspaces, target_mid, target_high)

    spec, log = coordinate_search(
        model=None, tok=None, dev_items=[object()], subspaces=subspaces, n_layers=n_layers,
        model_tag="Qwen2.5-7B-Instruct", eval_fn=stub,
    )
    # frozen spec's mid band is +0.4, high band is -0.2 (the toy targets)
    mid_vals = {v for v in spec.lam.values() if v > 0}
    high_vals = {v for v in spec.lam.values() if v < 0}
    assert mid_vals == {0.4}
    assert high_vals == {-0.2}
    # log has both sweep phases and its argmax rows match
    assert set(log["step"]) == {"lam_mid", "lam_high"}
    best_mid_row = log[log["step"] == "lam_mid"].sort_values("acc").iloc[-1]
    assert best_mid_row["lam_mid"] == 0.4


def test_coordinate_search_ablate_only_when_reinject_never_helps():
    """If negative λ_high only ever hurts, the search keeps λ_high = 0 (ablate-only)."""
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _toy_eval_factory(subspaces, target_mid=0.3, target_high=0.0)
    spec, _ = coordinate_search(
        model=None, tok=None, dev_items=[object()], subspaces=subspaces, n_layers=n_layers,
        model_tag="Qwen2.5-7B-Instruct", eval_fn=stub,
    )
    assert all(v > 0 for v in spec.lam.values())    # no re-injection band


# --- rank_sweep (stubbed run_eval; toy build_fn) -----------------------------------------------

def test_rank_sweep_argmax_full_curve_and_frozen_lam():
    d, n_layers = 16, 28
    ranks = (10, 15, 20, 30)
    target_rank = 20
    lam = make_schedule(6, 10, 0.3, n_layers, -0.2, n_layers=n_layers)

    # `current` publishes the rank being scored + its subspaces so the stub can read them back
    current: dict = {}

    def build_fn(r):
        subs = {l: _orthonormal(d, r, seed=1000 * r + l) for l in range(n_layers)}
        current["rank"], current["subspaces"] = r, subs
        return subs

    seen_lams = []

    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None,
             model_name=None, **kw):
        lams = {l: _recover_lam(fn, current["subspaces"][l]) for l, fn in hooks.items()}
        seen_lams.append(lams)
        acc = 1.0 - 0.01 * abs(current["rank"] - target_rank)   # tent peaked at target_rank
        return pd.DataFrame([{"model": "stub", "task": "mgsm", "lang": "en",
                              "correct": acc, "output_lang": "en"}])

    best_rank, curve = rank_sweep(
        model=None, tok=None, dev_items=[object()], build_fn=build_fn, lam=lam,
        n_layers=n_layers, ranks=ranks, eval_fn=stub,
    )
    assert best_rank == target_rank
    assert list(curve["rank"]) == list(ranks)            # full curve kept, never collapsed
    assert set(curve.columns) >= {"rank", "acc", "fidelity"}
    # λ was frozen across every rank (same schedule recovered each call)
    for lams in seen_lams:
        # round to 5 dp: _recover_lam back-solves λ from fp32 hook output, so ~1e-6 noise is
        # BLAS-dependent (observed on torch 2.13 CUDA builds) while real λs differ by ≥0.1
        pos = {round(v, 5) for v in lams.values() if v > 1e-9}
        neg = {round(v, 5) for v in lams.values() if v < -1e-9}
        assert pos == {0.3} and neg == {-0.2}


# --- make_bands --------------------------------------------------------------------------------

def test_make_bands_independent_bands_with_gap():
    sched = make_bands(ablate=(2, 5, 0.3), inject=(7, 10, -0.2), n_layers=12)
    assert all(sched[l] == 0.3 for l in (2, 3, 4))
    assert all(sched[l] == -0.2 for l in (7, 8, 9))
    assert all(l not in sched for l in (0, 1, 5, 6, 10, 11))   # gap (5,6) + tails untouched


def test_make_bands_clamps_to_range():
    sched = make_bands(ablate=(-2, 3, 0.4), inject=(10, 50, -0.1), n_layers=8)
    assert min(sched) >= 0 and max(sched) < 8


def test_make_bands_inject_wins_on_overlap():
    sched = make_bands(ablate=(2, 6, 0.3), inject=(4, 8, -0.2), n_layers=10)
    assert sched[4] == -0.2 and sched[5] == -0.2      # inject written last, wins
    assert sched[2] == 0.3 and sched[3] == 0.3


# --- pareto_front ------------------------------------------------------------------------------

def test_pareto_front_keeps_nondominated_and_ties():
    df = pd.DataFrame({
        "acc":      [0.9, 0.8, 0.7, 0.9],   # A best-acc, B best-fid, C dominated, D == A (tie)
        "fidelity": [0.2, 0.8, 0.1, 0.2],
    })
    mask = pareto_front(df)
    assert list(mask) == [True, True, False, True]


def test_pareto_front_nan_only_survives_by_leading_other_objective():
    df = pd.DataFrame({
        "acc":      [0.95, 0.80, 0.70],
        "fidelity": [float("nan"), 0.5, float("nan")],   # X leads acc -> stays; Z dominated by Y
    })
    mask = pareto_front(df)
    assert list(mask) == [True, True, False]


def test_pareto_front_edge_cases():
    assert list(pareto_front(pd.DataFrame({"acc": [0.5], "fidelity": [0.5]}))) == [True]
    assert len(pareto_front(pd.DataFrame({"acc": [], "fidelity": []}))) == 0


# --- grid_search: the joint λ argmax beats greedy on an interacting surface --------------------

def _interacting_eval_factory(subspaces):
    """Toy `run_eval` with a *deliberate* λ_mid×λ_high interaction: the accuracy-optimal λ_mid
    depends on λ_high (target λ_mid = 0.3 − λ_high), plus a re-injection bonus (−0.5·λ_high). Greedy
    coordinate descent fixes λ_mid at the λ_high=0 optimum (0.3) and never revisits it; the joint
    grid finds (λ_mid=0.5, λ_high=−0.3)."""
    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        lams = {l: _recover_lam(fn, subspaces[l]) for l, fn in hooks.items()}
        pos = [v for v in lams.values() if v > 1e-9]
        neg = [v for v in lams.values() if v < -1e-9]
        lam_mid = max(pos) if pos else 0.0
        lam_high = min(neg) if neg else 0.0
        acc = 1.0 - (lam_mid - (0.3 - lam_high)) ** 2 - 0.5 * lam_high
        return pd.DataFrame([{"model": "stub", "task": "mgsm", "lang": "en", "correct": acc}])
    return stub


def test_grid_search_joint_argmax_beats_greedy():
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _interacting_eval_factory(subspaces)
    kw = dict(model=None, tok=None, dev_items=[object()], subspaces=subspaces,
              n_layers=n_layers, model_tag="Qwen2.5-7B-Instruct", eval_fn=stub)

    g_spec, g_log = grid_search(**kw)
    c_spec, c_log = coordinate_search(**kw)

    # greedy gets stuck at λ_mid=0.3 (its λ_high=0 optimum); the joint grid moves to 0.5 / −0.3
    assert {v for v in c_spec.lam.values() if v > 0} == {0.3}
    assert {v for v in g_spec.lam.values() if v > 0} == {0.5}
    assert {v for v in g_spec.lam.values() if v < 0} == {-0.3}
    # and the joint optimum is strictly better on the dev metric
    assert g_log["acc"].max() > c_log["acc"].max()
    assert len(g_log) == 20                               # full 5×4 product scored


# --- optuna_search: searches layers, warm-starts from geometry, modular sampler ----------------

def _optuna_eval_factory(subspaces, target_center=8.0):
    """Toy two-objective `run_eval`: accuracy rewards an ablate band centered near `target_center`
    (so the search has a reason to leave the ρ-trough seed), fidelity rises with re-injection
    magnitude — a genuine accuracy/fidelity trade-off for the Pareto front."""
    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        lams = {l: _recover_lam(fn, subspaces[l]) for l, fn in hooks.items()}
        pos = [l for l, v in lams.items() if v > 1e-9]
        neg_mag = max((abs(v) for v in lams.values() if v < -1e-9), default=0.0)
        center = sum(pos) / len(pos) if pos else 0.0
        acc = max(0.0, 1.0 - abs(center - target_center) / 12.0)
        k = round(4 * min(1.0, neg_mag / 0.3))
        rows = [{"model": "stub", "task": "mgsm", "lang": "en", "correct": acc,
                 "output_lang": "en" if i < k else "xx"} for i in range(4)]
        return pd.DataFrame(rows)
    return stub


def test_optuna_search_warmstart_explores_and_returns_pareto_specs():
    from intervene import InterventionSpec
    d, r, n_layers = 16, 3, 12
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _optuna_eval_factory(subspaces)

    study, specs = optuna_search(
        model=None, tok=None, dev_items=[object()], subspaces=subspaces, n_layers=n_layers,
        model_tag="Test-Model", n_trials=12, eval_fn=stub, seed=0,   # unlisted -> trough=round(0.6*12)=7
    )

    # multi-objective, non-empty Pareto front of real specs
    assert len(study.directions) == 2
    assert len(specs) >= 1 and all(isinstance(s, InterventionSpec) for s in specs)
    # geometry warm-start is trial 0 (place_bands(7,12) -> ablate [5,9), inject [9,12))
    assert study.trials[0].params == {"a_start": 5, "a_end": 9, "i_start": 9, "i_end": 12,
                                      "lam_mid": 0.4, "lam_high": -0.3}
    # the layer axis is genuinely searched, not frozen at the trough seed
    assert len({t.params["a_start"] for t in study.trials}) > 1


def test_optuna_search_accepts_injected_sampler():
    import optuna
    d, r, n_layers = 16, 3, 12
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _optuna_eval_factory(subspaces)
    study, specs = optuna_search(
        model=None, tok=None, dev_items=[object()], subspaces=subspaces, n_layers=n_layers,
        model_tag="Test-Model", n_trials=8, eval_fn=stub,
        sampler=optuna.samplers.RandomSampler(seed=1),
    )
    assert len(specs) >= 1


# --- layer_sweep (Zhao Fig 4) + strength_sweep (Zhao Fig 3) ---------------------------------------

def test_layer_sweep_full_curve_and_fidelity_columns():
    d, n_layers = 16, 28
    subs = {l: _orthonormal(d, 10, seed=l) for l in range(n_layers)}
    target_start = 8

    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        start = min(hooks) if hooks else -1            # ablate band start = lowest hooked layer
        acc = 1.0 - 0.01 * abs(start - target_start)   # tent peaked at target_start
        return pd.DataFrame([{"model": "s", "task": "mgsm", "lang": "en",
                              "correct": acc, "output_lang": "en"}])

    best_start, curve = layer_sweep(
        None, None, [object()], subs, n_layers, lam_mid=0.3, lam_high=0.0,
        band_width=4, starts=[0, 4, 8, 12, 16], eval_fn=stub,
    )
    assert best_start == target_start
    assert list(curve["start"]) == [0, 4, 8, 12, 16]                 # full curve kept
    assert set(curve.columns) >= {"start", "acc", "fidelity", "reasoning_fidelity"}


def test_strength_sweep_signed_axis_with_clean_zero():
    d, n_layers = 16, 28
    subs = {l: _orthonormal(d, 10, seed=l) for l in range(n_layers)}
    seen_hooks = []

    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        seen_hooks.append(len(hooks) if hooks else 0)
        return pd.DataFrame([{"model": "s", "task": "mgsm", "lang": "en",
                              "correct": 1.0, "output_lang": "en"}])

    strengths = (-0.2, 0.0, 0.2)
    log = strength_sweep(None, None, [object()], subs, n_layers, trough=14,
                         strengths=strengths, eval_fn=stub)
    assert list(log["strength"]) == list(strengths)
    assert set(log.columns) >= {"strength", "acc", "fidelity", "reasoning_fidelity"}
    # strength 0 => empty schedule => clean (no hooks); non-zero strengths edit the band
    assert seen_hooks[strengths.index(0.0)] == 0
    assert all(seen_hooks[i] > 0 for i, s in enumerate(strengths) if s != 0.0)


def test_lam_sweep_symmetric_spans_signs_and_zero():
    assert 0.0 in LAM_SWEEP_SYMMETRIC
    assert min(LAM_SWEEP_SYMMETRIC) < 0 < max(LAM_SWEEP_SYMMETRIC)   # inject (−) through ablate (+)


def test_rank_sweep_logs_dose_when_dose_fn_given():
    d, n_layers = 16, 28
    lam = make_schedule(6, 10, 0.3, n_layers, -0.2, n_layers)
    def build_fn(r): return {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        return pd.DataFrame([{"model": "s", "task": "mgsm", "lang": "en",
                              "correct": 0.5, "output_lang": "en"}])
    _, curve = rank_sweep(None, None, [object()], build_fn, lam, n_layers,
                          ranks=(10, 20), eval_fn=stub, dose_fn=lambda subs: 0.01 * len(subs))
    assert "dose" in curve.columns and (curve["dose"] > 0).all()


# --- Group A additions: lam_from_params, boundary flag, keep_items, select_rank_1se, provenance ---

def test_lam_from_params_public_and_alias():
    from tuning import _lam_from_params
    params = {"a_start": 2, "a_end": 5, "i_start": 7, "i_end": 10, "lam_mid": 0.3, "lam_high": -0.2}
    sched = lam_from_params(params, n_layers=12, per_layer=False)
    assert sched[2] == 0.3 and sched[4] == 0.3 and sched[7] == -0.2 and sched[9] == -0.2
    assert lam_from_params is _lam_from_params                    # private name still importable


def test_coordinate_search_flags_boundary_at_dose_ceiling():
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    # toy argmax at the ablation ceiling (λ_mid=0.5=grid max) + re-injection ceiling (λ_high=-0.3=grid min)
    stub = _toy_eval_factory(subspaces, target_mid=0.5, target_high=-0.3)
    _, log = coordinate_search(model=None, tok=None, dev_items=[object()], subspaces=subspaces,
                               n_layers=n_layers, model_tag="Qwen2.5-7B-Instruct", eval_fn=stub)
    assert log.attrs["boundary"] == {"lam_mid": True, "lam_high": True}


def test_coordinate_search_no_boundary_for_interior_and_ablate_only():
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    # interior λ_mid=0.3, and λ_high=0.0 (ablate-only) is the *floor*, not the re-injection ceiling
    stub = _toy_eval_factory(subspaces, target_mid=0.3, target_high=0.0)
    _, log = coordinate_search(model=None, tok=None, dev_items=[object()], subspaces=subspaces,
                               n_layers=n_layers, model_tag="Qwen2.5-7B-Instruct", eval_fn=stub)
    assert log.attrs["boundary"] == {"lam_mid": False, "lam_high": False}


# --- boundary-triggered escalation (E7 Q&A Q5): widen only the pinned side, clamped to caps ------

def test_coordinate_search_escalates_past_pinned_grid_edges():
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    # true optimum outside BOTH default grids (mid ceiling 0.5, high floor -0.3): the base sweep pins
    # at the edges, one escalation pass extends mid by (0.75, 1.0) and high by (-0.45, -0.6)
    stub = _toy_eval_factory(subspaces, target_mid=0.75, target_high=-0.45)
    spec, log = coordinate_search(model=None, tok=None, dev_items=[object()], subspaces=subspaces,
                                  n_layers=n_layers, model_tag="Qwen2.5-7B-Instruct", eval_fn=stub,
                                  escalate=1)
    assert {v for v in spec.lam.values() if v > 0} == {0.75}
    assert {v for v in spec.lam.values() if v < 0} == {-0.45}
    assert log.attrs["escalation"]["passes"] == 1
    assert max(log.attrs["escalation"]["lam_mid_grid"]) == 1.0
    assert min(log.attrs["escalation"]["lam_high_grid"]) == -0.6
    assert log.attrs["boundary"] == {"lam_mid": False, "lam_high": False}   # interior after widening
    assert {"lam_mid_esc", "lam_high_esc"} <= set(log["step"])


def test_coordinate_search_escalation_clamps_at_cap_and_stops():
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _toy_eval_factory(subspaces, target_mid=2.0, target_high=0.0)   # wants far more ablation
    # cap 0.6 < the 1.5x/2x continuation: the single extension point is the cap itself, the argmax
    # stays pinned there, and a second allowed pass cannot grow (break branch) — passes stays 1
    spec, log = coordinate_search(model=None, tok=None, dev_items=[object()], subspaces=subspaces,
                                  n_layers=n_layers, model_tag="Qwen2.5-7B-Instruct", eval_fn=stub,
                                  escalate=2, lam_mid_cap=0.6)
    assert {v for v in spec.lam.values() if v > 0} == {0.6}
    assert log.attrs["escalation"]["passes"] == 1
    assert max(log.attrs["escalation"]["lam_mid_grid"]) == 0.6
    assert log.attrs["boundary"]["lam_mid"] is True     # honest: still pinned, at the physical cap


def test_coordinate_search_escalate_default_off_is_old_behavior():
    d, r, n_layers = 16, 4, 28
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _toy_eval_factory(subspaces, target_mid=0.75, target_high=-0.45)
    spec, log = coordinate_search(model=None, tok=None, dev_items=[object()], subspaces=subspaces,
                                  n_layers=n_layers, model_tag="Qwen2.5-7B-Instruct", eval_fn=stub)
    assert {v for v in spec.lam.values() if v > 0} == {0.5}                 # pinned at the old ceiling
    assert len(log) == 9                                                    # 5 mids + 4 highs, no extras
    assert log.attrs["escalation"]["passes"] == 0
    assert log.attrs["boundary"] == {"lam_mid": True, "lam_high": True}


def _pinning_optuna_eval_factory(subspaces):
    """Monotone-in-λ_mid scorer: the enqueued geometry seed (λ_mid = range top when the range tops
    below 0.4) is deterministically the accuracy champion, so the boundary flag must fire."""
    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        lams = {l: _recover_lam(fn, subspaces[l]) for l, fn in hooks.items()}
        lam_mid = max([v for v in lams.values() if v > 1e-9], default=0.0)
        return pd.DataFrame([{"model": "stub", "task": "mgsm", "lang": "en",
                              "correct": min(1.0, lam_mid), "output_lang": "en"}])
    return stub


def test_optuna_search_escalates_pinned_range_and_merges_passes():
    d, r, n_layers = 16, 3, 12
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _pinning_optuna_eval_factory(subspaces)
    # seed λ_mid = clamp(0.4 into (0.1, 0.4)) = 0.4 = the range top; seed λ_high = -0.3 = the range
    # floor -> both sides flagged, both widened by one span (mid 0.4->0.7, high -0.3->-0.6)
    study, specs = optuna_search(
        model=None, tok=None, dev_items=[object()], subspaces=subspaces, n_layers=n_layers,
        model_tag="Test-Model", n_trials=8, eval_fn=stub, seed=0,
        lam_mid_range=(0.1, 0.4), lam_high_range=(-0.3, 0.0), escalate=1,
    )
    esc = study.user_attrs["escalation"]
    assert esc["passes"] == 1
    assert esc["lam_mid_range"] == [0.1, 0.7] and esc["lam_high_range"] == [-0.6, 0.0]
    assert len(study.trials) == 16                       # pass-2 trials + pass-1 merged back in
    champ = max(study.best_trials, key=lambda t: t.values[0])
    assert champ.params["lam_mid"] >= 0.4 - 1e-9         # never worse than the pinned champion
    assert len(specs) >= 1
    # Group-G provenance still points at the TRUE geometry seed, not the escalation warm-start
    from tuning import optuna_warmstart_delta
    prov = optuna_warmstart_delta(study, champ)
    assert (prov["warm_a_start"], prov["warm_a_end"], prov["warm_i_start"], prov["warm_i_end"]) == (5, 9, 9, 12)
    assert abs(prov["warm_acc"] - 0.4) < 1e-6            # the seed's own dev acc, preserved through the merge (fp32 hooks)


def test_optuna_search_escalate_default_off_records_zero_passes():
    d, r, n_layers = 16, 3, 12
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _pinning_optuna_eval_factory(subspaces)
    study, _ = optuna_search(
        model=None, tok=None, dev_items=[object()], subspaces=subspaces, n_layers=n_layers,
        model_tag="Test-Model", n_trials=8, eval_fn=stub, seed=0,
        lam_mid_range=(0.1, 0.4), lam_high_range=(-0.3, 0.0),
    )
    assert study.user_attrs["escalation"]["passes"] == 0
    assert study.user_attrs["escalation"]["lam_mid_range"] == [0.1, 0.4]
    assert len(study.trials) == 8
    assert study.user_attrs["boundary"]["lam_mid"] is True   # flagged but not acted on


def test_rank_sweep_keep_items_captures_per_item_ids_and_correct():
    d, n_layers = 16, 28
    lam = make_schedule(6, 10, 0.3, n_layers, -0.2, n_layers)
    def build_fn(r): return {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        return pd.DataFrame([{"model": "s", "task": "mgsm", "lang": "en", "id": "p-0", "correct": 1},
                             {"model": "s", "task": "mgsm", "lang": "en", "id": "p-1", "correct": 0}])
    _, curve = rank_sweep(None, None, [object()], build_fn, lam, n_layers,
                          ranks=(10, 20), eval_fn=stub, keep_items=True)
    assert set(curve.attrs["items"]) == {10, 20}
    assert curve.attrs["items"][10]["ids"] == ["p-0", "p-1"]
    assert curve.attrs["items"][10]["correct"] == [1, 0]
    # default path leaves attrs["items"] absent (byte-identical behavior)
    _, curve2 = rank_sweep(None, None, [object()], build_fn, lam, n_layers, ranks=(10,), eval_fn=stub)
    assert "items" not in curve2.attrs


def test_rank_sweep_retuned_retunes_per_rank_and_splits_dev_test():
    # E7 change #3: λ is RE-TUNED per rank (equal footing), curve rows are TEST metrics while
    # attrs["items"] holds the DEV per-item correctness (the select_rank_1se substrate) — no leakage.
    d, n_layers = 16, 28
    ranks = (5, 10, 20)
    dev_items = [("dev", i) for i in range(2)]
    test_items = [("test", i) for i in range(2)]
    seen = []                                                     # which split each eval scored

    class _Spec:                                                  # minimal (spec.lam, spec.scope) contract
        def __init__(self, lam, scope="prompt_only"):
            self.lam, self.scope = lam, scope

    def build_fn(r):
        return {l: _orthonormal(d, r, seed=1000 * r + l) for l in range(n_layers)}

    def tune_fn(subs, dev):
        assert dev is dev_items                                   # λ search sees DEV only
        r = int(next(iter(subs.values())).shape[1])               # recover rank from the subspace
        return _Spec({l: 0.1 * r for l in range(3)}), None        # a schedule that VARIES with rank

    def stub(model, tok, items, *, hooks=None, mask_holder=None, device=None, model_name=None, **kw):
        seen.append("test" if items is test_items else "dev" if items is dev_items else "other")
        acc = 0.9 if items is test_items else 0.5                 # distinct so we can tell them apart
        return pd.DataFrame([{"model": "s", "task": "mgsm", "lang": "en", "id": "p-0",
                              "correct": acc, "output_lang": "en"}])

    best_rank, curve = rank_sweep_retuned(
        None, None, dev_items, test_items, build_fn, n_layers,
        tune_fn=tune_fn, ranks=ranks, eval_fn=stub,
        dose_fn=lambda lam: (lambda subs: 0.01 * sum(lam.values())), keep_items=True,
    )
    assert list(curve["rank"]) == list(ranks)                     # full curve, never collapsed
    assert "lam" in curve.columns and curve["lam"].nunique() == len(ranks)   # λ retuned per rank
    assert (curve["acc"] == 0.9).all()                            # curve reads TEST (0.9), not dev (0.5)
    assert "dose" in curve.columns and (curve["dose"] > 0).all()  # dose via the FACTORY dose_fn(lam)(subs)
    assert set(curve.attrs["items"]) == set(ranks)                # DEV per-item substrate for r* selection
    assert seen.count("test") == len(ranks) and seen.count("dev") == len(ranks)  # both splits scored per rank


def test_select_rank_1se_prefers_smaller_rank_within_one_se():
    # deterministic homogeneous clusters (both parallel rows agree) so effective n ≈ K=100
    clusters = np.repeat(np.arange(100), 2)
    correct = {
        10: np.array([1] * 120 + [0] * 80),      # acc 0.60
        20: np.array([1] * 124 + [0] * 76),      # acc 0.62  <- argmax
        50: np.array([1] * 110 + [0] * 90),      # acc 0.55
    }
    clus = {r: clusters for r in correct}
    r_star, r_argmax, se = select_rank_1se(correct, clus, n_boot=400, seed=1)
    assert r_argmax == 20                          # highest pooled dev accuracy
    assert r_star == 10                            # 0.60 is within 1 cluster-SE of 0.62 -> parsimony
    assert se > 0


def test_optuna_search_sets_boundary_user_attr_and_warmstart_delta():
    d, r, n_layers = 16, 3, 12
    subspaces = {l: _orthonormal(d, r, seed=l) for l in range(n_layers)}
    stub = _optuna_eval_factory(subspaces)
    study, specs = optuna_search(
        model=None, tok=None, dev_items=[object()], subspaces=subspaces, n_layers=n_layers,
        model_tag="Test-Model", n_trials=12, eval_fn=stub, seed=0,
    )
    assert set(study.user_attrs["boundary"]) == {"lam_mid", "lam_high"}
    champ = max(study.best_trials, key=lambda t: t.values[0])
    prov = optuna_warmstart_delta(study, champ)
    assert set(prov) >= {"warm_a_start", "warm_a_end", "chosen_a_start", "band_shift",
                         "left_trough", "warm_acc", "chosen_acc"}
    assert prov["warm_a_start"] == 5 and prov["warm_a_end"] == 9   # place_bands(7,12) trough seed
    assert isinstance(prov["left_trough"], bool)
