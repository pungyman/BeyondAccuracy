"""Dose accounting: the common scale on which intervention strengths are comparable.

A schedule's **dose** is its total edit energy relative to the residual stream,

    dose(lam) = sum_l lam_l^2 * e_l,     e_l = E||delta_l(x)||^2 / E||x||^2   (`operators.edit_energy`)

`e_l` is the unit-lam energy fraction at layer `l`, so raw `lam` values are NOT comparable across
arms -- operators differ by 10-40x in `e_l` at equal `lam` (E6 "dose comparability"). Dose is.

Two consequences the E9 trade-off grid is built on:

1. **Dose is quadratic in a uniform rescale.** `dose(alpha*lam) = alpha^2 * dose(lam)` exactly, and
   scaling every layer by one `alpha` preserves the ablate/re-inject band ratio (the sign structure
   of `intervene.InterventionSpec.lam`). So hitting a dose target is a closed-form solve,
   `alpha = sqrt(target / measured)` -- no search, no generation.
2. **`e_l` is measured per language.** The edit itself is language-agnostic (fit on the pooled
   parallel corpus), but the activations it acts on are not, so one schedule delivers a different
   dose in bn than in de. That difference is exactly what a per-language dose calibration removes.

These helpers previously lived inside `scripts/e6_phase2_behavioral.py`, which loads a model at
module level and has no `__main__` guard -- nothing there is importable. They are here so
`e9_dose_calibrate.py`, the E6 ladder and the searchlog annotation share one definition, and so the
per-language variant exists at all (the E6 table pools all languages into one `e_l`).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from intervene import AffineEdit
from operators import edit_energy

#: `alpha` clamp for the dose solve. A cell that clamps did NOT reach its target and must be
#: reported as such -- outside this range the rescale stops being a strength knob and starts being
#: a different intervention (E6 used the same 0.3/3.0 bounds).
ALPHA_MIN, ALPHA_MAX = 0.3, 3.0


def _rows(n: int, n_rows: int | None, seed: int) -> Tensor | slice:
    """Seeded row subsample of `n` rows, or every row when `n_rows` covers them."""
    if n_rows is None or n_rows >= n:
        return slice(None)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[:n_rows]


def edit_energy_by_lang(edits: dict[int, AffineEdit], acts: dict[str, Tensor],
                        langs=None, *, n_rows: int | None = 512, seed: int = 0
                        ) -> dict[tuple[int, str], float]:
    """`{(layer, lang) -> e_l}` — the unit-lam edit energy of `edits` on each language's own rows.

    `acts` is the `{lang: (n, n_layers, d)}` activation cache; only the layers present in `edits`
    are touched. `e_l` is an expectation over rows, so a seeded `n_rows` subsample is enough and
    keeps the sweep cheap at n=7500 x 11 languages (set `n_rows=None` for every row).

    The per-language split is the point: `edits` is one language-agnostic map, but `E||x||^2` and
    the edit's overlap with the stream both vary by language, so `e_l` does too.
    """
    langs = list(acts) if langs is None else list(langs)
    out: dict[tuple[int, str], float] = {}
    for lang in langs:
        a = acts[lang]
        sel = _rows(int(a.shape[0]), n_rows, seed)
        for layer, edit in edits.items():
            out[(int(layer), lang)] = edit_energy(edit, a[sel, int(layer), :])
    return out


def schedule_dose(lam: dict[int, float], e_by_layer: dict[int, float]) -> float:
    """`sum_l lam_l^2 * e_l` over the layers the schedule actually edits.

    Layers with `lam == 0` contribute nothing (and `intervene.build_affine_hooks` drops them), so
    the sum is over `lam`'s support. Raises if a schedule edits a layer with no measured `e_l` —
    silently scoring that layer as zero dose would understate the edit.
    """
    total = 0.0
    for layer, l in lam.items():
        if l == 0.0:
            continue
        layer = int(layer)
        if layer not in e_by_layer:
            raise KeyError(f"schedule edits layer {layer} at lam={l} but no e_l was measured there "
                           f"(have {sorted(e_by_layer)})")
        total += float(l) ** 2 * float(e_by_layer[layer])
    return total


def scale_schedule(lam: dict[int, float], alpha: float) -> dict[int, float]:
    """Scale every layer's strength by `alpha`, preserving the signed band structure.

    Both bands scale together, so the ablate:re-inject ratio — and therefore the *shape* of the
    intervention — is unchanged; only its size moves. This is what makes `dose(alpha*lam) =
    alpha^2 * dose(lam)` a strength knob rather than a new arm.
    """
    return {int(layer): float(alpha) * float(l) for layer, l in lam.items()}


def solve_alpha(dose_target: float, dose_at_lam: float, *,
                lo: float = ALPHA_MIN, hi: float = ALPHA_MAX) -> tuple[float, bool]:
    """`(alpha, clamped)` such that `dose(alpha*lam) ~= dose_target`, from `alpha = sqrt(t/d)`.

    Closed form, because dose is quadratic in a uniform rescale. `clamped` is True when the exact
    solution fell outside `[lo, hi]` and was pinned — the caller MUST surface that, since a clamped
    cell sits at a dose it was not asked for.
    """
    if not (dose_at_lam > 0.0):
        raise ValueError(f"cannot solve a dose target against dose_at_lam={dose_at_lam!r}; "
                         f"a zero/negative measured dose means the schedule edits nothing")
    if dose_target < 0.0:
        raise ValueError(f"dose_target must be >= 0, got {dose_target!r}")
    exact = float(np.sqrt(dose_target / dose_at_lam))
    alpha = float(min(max(exact, lo), hi))
    return alpha, bool(alpha != exact)


def pool_dose(doses, how: str = "geo") -> float:
    """Pool per-language doses into the one cross-language anchor a dose rung is measured against.

    `"geo"` (default) is the geometric mean: dose is an energy *ratio*, so the pooling that treats
    a 2x-over and a 2x-under language symmetrically is the one in log space. `"mean"` is the plain
    arithmetic mean, for when the anchor should track the largest edits.
    """
    v = np.asarray([float(x) for x in doses], dtype=np.float64)
    if v.size == 0 or not np.all(v > 0):
        raise ValueError(f"need strictly positive doses to pool, got {v.tolist()}")
    if how == "geo":
        return float(np.exp(np.log(v).mean()))
    if how == "mean":
        return float(v.mean())
    raise ValueError(f"unknown pooling {how!r} (want 'geo' or 'mean')")
