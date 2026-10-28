"""Unit tests for the dose scale and the closed-form dose->lambda solve (`src/dose.py`).

The load-bearing claims, all on synthetic planted data:

- `e_l` is measured per LANGUAGE, and genuinely differs across languages for one shared edit —
  which is the entire reason a per-language calibration exists.
- The solve round-trips: `schedule_dose(scale_schedule(lam, solve_alpha(target, d0)))` == target,
  with the dose RECOMPUTED from the scaled schedule rather than derived as `alpha^2 * d0`, so the
  test would catch a broken `scale_schedule` that the algebraic identity would hide.
- A clamped solve is reported as clamped. A cell pinned at the alpha bound sits at a dose it was
  not asked for, and E9 must be able to mark it rather than plot it as if it hit its rung.
- `scale_schedule` preserves the signed ablate/re-inject band ratio (it is a strength knob, not a
  different arm).
"""
import numpy as np
import pytest
import torch

from dose import (ALPHA_MAX, ALPHA_MIN, edit_energy_by_lang, pool_dose,
                  scale_schedule, schedule_dose, solve_alpha)
from operators import build_family_edits
from tuning import paper_fixed_schedule

from test_operators import _planted


def _cache(n_layers=12, **kw):
    acts, _, _ = _planted(d=24, r=3, L=4, n=60, noise=0.05, **kw)
    return {l: a.unsqueeze(1).repeat(1, n_layers, 1) for l, a in acts.items()}


def _energies(cache, lam, basis="alg1", operator="proj", rank=3):
    layers = sorted(lam)
    edits = build_family_edits(cache, layers, basis=basis, operator=operator, rank=rank)
    return layers, edit_energy_by_lang(edits, cache, n_rows=None)


# --- per-language energy ---------------------------------------------------------------------------

def test_edit_energy_is_per_language_and_differs():
    cache = _cache(lang_scale=2.0)
    lam = paper_fixed_schedule(12)
    layers, E = _energies(cache, lam)
    assert set(E) == {(l, lg) for l in layers for lg in cache}
    per_lang = [E[(layers[0], lg)] for lg in cache]
    assert all(e > 0 for e in per_lang)
    # one shared, language-agnostic edit still lands a different energy on each language's rows
    assert max(per_lang) > 1.05 * min(per_lang)


def test_edit_energy_subsample_is_seeded_and_close_to_full():
    cache = _cache()
    lam = paper_fixed_schedule(12)
    layers = sorted(lam)
    edits = build_family_edits(cache, layers[:1], basis="alg1", operator="proj", rank=3)
    a = edit_energy_by_lang(edits, cache, n_rows=30, seed=0)
    b = edit_energy_by_lang(edits, cache, n_rows=30, seed=0)
    full = edit_energy_by_lang(edits, cache, n_rows=None)
    assert a == b                                          # seeded -> reproducible
    for k in full:
        assert a[k] == pytest.approx(full[k], rel=0.5)     # an expectation, so a subsample tracks it


# --- the solve -------------------------------------------------------------------------------------

def test_schedule_dose_ignores_zero_lam_and_raises_on_unmeasured_layer():
    e = {4: 0.1, 5: 0.2}
    assert schedule_dose({4: 0.0, 5: 0.5}, e) == pytest.approx(0.25 * 0.2)
    with pytest.raises(KeyError):
        schedule_dose({9: 0.3}, e)                         # edits a layer with no measured e_l


def test_scale_schedule_preserves_the_signed_band_ratio():
    lam = paper_fixed_schedule(12)
    scaled = scale_schedule(lam, 1.7)
    assert set(scaled) == set(lam)
    for l in lam:
        assert np.sign(scaled[l]) == np.sign(lam[l])
        assert scaled[l] == pytest.approx(1.7 * lam[l])
    mid = [l for l, v in lam.items() if v > 0][0]
    high = [l for l, v in lam.items() if v < 0][0]
    assert scaled[mid] / scaled[high] == pytest.approx(lam[mid] / lam[high])


@pytest.mark.parametrize("k", [0.5, 1.0, 1.5, 2.0])
def test_solve_round_trips_to_the_dose_target_per_language(k):
    cache = _cache()
    lam = paper_fixed_schedule(12)
    layers, E = _energies(cache, lam)
    d0 = {lg: schedule_dose(lam, {l: E[(l, lg)] for l in layers}) for lg in cache}
    target = k * pool_dose(d0.values(), "geo")
    alphas = {}
    for lg in cache:
        alpha, clamped = solve_alpha(target, d0[lg])
        assert not clamped
        alphas[lg] = alpha
        # recomputed from the scaled schedule, NOT alpha**2 * d0
        measured = schedule_dose(scale_schedule(lam, alpha), {l: E[(l, lg)] for l in layers})
        assert measured == pytest.approx(target, rel=1e-9)
    # a common absolute target on languages of differing dose => genuinely per-language lambda
    assert max(alphas.values()) > min(alphas.values())


def test_solve_alpha_clamps_and_says_so():
    assert solve_alpha(1.0, 1.0) == (1.0, False)
    hi, hi_clamped = solve_alpha(1e6, 1.0)
    lo, lo_clamped = solve_alpha(1e-9, 1.0)
    assert (hi, hi_clamped) == (ALPHA_MAX, True)
    assert (lo, lo_clamped) == (ALPHA_MIN, True)
    with pytest.raises(ValueError):
        solve_alpha(1.0, 0.0)                              # nothing to scale
    with pytest.raises(ValueError):
        solve_alpha(-1.0, 1.0)


def test_pool_dose_geo_is_log_symmetric():
    assert pool_dose([1.0, 4.0], "geo") == pytest.approx(2.0)
    assert pool_dose([1.0, 4.0], "mean") == pytest.approx(2.5)
    assert pool_dose([2.0], "geo") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        pool_dose([1.0, 0.0], "geo")
    with pytest.raises(ValueError):
        pool_dose([1.0], "median")
