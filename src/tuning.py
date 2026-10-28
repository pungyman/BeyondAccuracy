"""E5 lever-2 search: the per-layer λ schedule, geometry-seeded and dev-tuned.

Implements [hyperparameter_search.md](../docs/experiments/hyperparameter_search.md) lever 2 — the
two-band, four-number λ schedule (mid-ablate `λ>0`, high-reinject `λ≤0`) and a coordinate/greedy
search over it on the **dev** split only (the data-role discipline of E5 §2). Everything here is
subspace-agnostic: it takes a `{layer -> M_s}` dict, so P3 reuses it unchanged for DiffPCA.

Band *placement* is seeded from each model's E3 mid-network ρ trough (`TROUGH_LAYER`), not guessed
— turning the paper's fixed layer indices into a per-model, geometry-informed choice. The paper's
own fixed schedule is exposed as `paper_fixed_schedule` so "our geometry-placed schedule beats the
fixed one" (H4) is a measurable claim.

E6 note: every search here also accepts a `{layer -> operators.AffineEdit}` mapping in place of
`{layer -> M_s}` (detected by value type) and then freezes an `AffineSpec` / builds hooks via
`build_affine_hooks` — the λ-schedule semantics (bands, signs, scope) are identical, so the
Plan-B operators ride the same searches unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from intervene import (AffineEdit, AffineSpec, InterventionSpec, MaskHolder,
                       build_affine_hooks, build_layer_hooks)

# E3 mid-network ρ troughs (docs/experiments/E3_results.md, H2): the layer where language and
# content are maximally entangled — the ablate band is centered here. Models not listed fall back
# to ~0.6·n_layers (the ~60–70% relative depth the trough sits at across the ladder).
TROUGH_LAYER: dict[str, int] = {
    "Qwen2.5-1.5B-Instruct": 16,
    "Qwen2.5-3B-Instruct": 25,
    "Qwen2.5-7B-Instruct": 18,
}

# Strength grids (hyperparameter_search.md lever 2, paper-adjacent). λ_mid>0 ablates; λ_high≤0
# re-injects (λ_high=0 is the ablate-only ablation of the re-injection idea itself).
LAM_MID_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
LAM_HIGH_GRID: tuple[float, ...] = (0.0, -0.1, -0.2, -0.3)

# Signed single-band strength grid for the Zhao Fig-3 axis ("proportion removed (+) / injected (−)"):
# one ablate band swept from re-injection through clean (0) to full ablation. `strength_sweep` uses it.
LAM_SWEEP_SYMMETRIC: tuple[float, ...] = (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)

# Rank grid for lever 1 / H3 (hyperparameter_search.md): the accuracy-vs-rank curve is swept over
# these ranks — the E7 unified sweep re-tunes λ per rank (`rank_sweep_retuned`), so the grid spans
# BOTH sides of r=10. r=10 (=cfg.rank, the Alg1/`none` structural cap) is the rank-matched head-to-head
# operating point; ranks below probe whether DiffPCA peaks under the L-1 cap, ranks above whether its
# extra dimensions buy accuracy (the E5 §7 rank-parity control keeps the full curve, never just r*).
RANK_GRID: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100)

# The paper's `r1-7b_mgsm.sh` reference schedule on a 28-layer model: ablate layers 10–19 @0.3,
# re-inject layers 20–27 @−0.3. `paper_fixed_schedule` scales these fractions to any depth.
_PAPER_REF_LAYERS = 28
_PAPER_MID = (10, 20)          # [start, end) -> layers 10..19 ablate
_PAPER_HIGH_START = 20         # [20, n_layers) -> reinject to the head
_PAPER_LAM_MID = 0.3
_PAPER_LAM_HIGH = -0.3


_TROUGH_LOGGED: set[str] = set()


def _resolve_trough(model_tag: str, n_layers: int) -> tuple[int, str]:
    """Trough + provenance. Precedence: E5_TROUGH → E5_TROUGH_JSON → dict → 0.6·depth.

    The env channels exist so a driver run can seed bands from a freshly *measured* trough
    (scripts/extract_trough.py writes the JSON) without changing the behavior of any process
    that doesn't set them — the dict/fallback path below is byte-identical to the historical
    resolution, so prior runs stay reproducible.
    """
    if os.environ.get("E5_TROUGH"):
        v = int(os.environ["E5_TROUGH"])
        if not 0 <= v < n_layers:
            raise ValueError(f"E5_TROUGH={v} out of range for {model_tag} (n_layers={n_layers})")
        return v, "env E5_TROUGH"
    jpath = os.environ.get("E5_TROUGH_JSON")
    if jpath and Path(jpath).exists():
        meta = json.loads(Path(jpath).read_text())
        v = int(meta.get("trough_layer", -1))
        if meta.get("model_tag") == model_tag and 0 <= v < n_layers:
            return v, f"measured {Path(jpath).name} (n={meta.get('n')})"
        print(f"[tuning] WARNING: E5_TROUGH_JSON={jpath} does not match {model_tag} "
              f"(tag={meta.get('model_tag')}, trough={meta.get('trough_layer')}); ignoring")
    if model_tag in TROUGH_LAYER:
        return TROUGH_LAYER[model_tag], "TROUGH_LAYER dict (n=512)"
    return round(0.6 * n_layers), "0.6*depth fallback"


def trough_layer(model_tag: str, n_layers: int) -> int:
    """This model's E3 ρ trough (band seed); ~0.6·n_layers fallback for unlisted models."""
    val, src = _resolve_trough(model_tag, n_layers)
    if model_tag not in _TROUGH_LOGGED:  # searches call this per candidate; log once per tag
        print(f"[tuning] trough({model_tag}, L={n_layers}) = {val}  [source: {src}]", flush=True)
        _TROUGH_LOGGED.add(model_tag)
    return val


def make_schedule(
    mid_start: int, mid_end: int, lam_mid: float, high_end: int, lam_high: float, n_layers: int
) -> dict[int, float]:
    """Two-band per-layer λ map: `lam_mid` on `[mid_start, mid_end)`, `lam_high` on
    `[mid_end, high_end)`.

    The four-number parameterization of hyperparameter_search.md lever 2 (the high band starts
    where the mid band ends). Layers are clamped to `[0, n_layers)`; entries whose λ is exactly 0
    are dropped, so an ablate-only sweep (`lam_high=0`) yields a mid-band-only schedule.
    """
    sched: dict[int, float] = {}
    for layer in range(max(0, mid_start), min(n_layers, mid_end)):
        if lam_mid != 0.0:
            sched[layer] = lam_mid
    for layer in range(max(0, mid_end), min(n_layers, high_end)):
        if lam_high != 0.0:
            sched[layer] = lam_high
    return sched


def make_bands(
    ablate: tuple[int, int, float], inject: tuple[int, int, float], n_layers: int
) -> dict[int, float]:
    """Two **independent** per-layer bands with an optional untouched gap between them.

    `ablate=(start, end, lam)` sets `lam` on `[start, end)`; `inject=(start, end, lam)` likewise on
    its own `[start, end)`. Unlike `make_schedule` — whose high band is *pinned* to start where the
    mid band ends — the two bands here are placed independently, so `optuna_search` can relocate the
    ablate and re-inject layers separately (and leave a gap of untouched layers between them) instead
    of taking the E3 ρ trough for granted. Layers are clamped to `[0, n_layers)`; exactly-0 λ entries
    are dropped; on any overlap the inject band wins (written last).
    """
    a_start, a_end, lam_a = ablate
    i_start, i_end, lam_i = inject
    sched: dict[int, float] = {}
    for layer in range(max(0, a_start), min(n_layers, a_end)):
        if lam_a != 0.0:
            sched[layer] = lam_a
    for layer in range(max(0, i_start), min(n_layers, i_end)):
        if lam_i != 0.0:
            sched[layer] = lam_i
    return sched


def place_bands(trough: int, n_layers: int, *, half_width: int | None = None) -> tuple[int, int, int]:
    """`(mid_start, mid_end, high_end)`: an ablate band centered on `trough`, re-inject above it
    to `L_max`.

    `half_width` defaults to ~0.15·n_layers (≈4 layers for a 28-layer model), matching the paper's
    ~10-layer ablate band. All three edges are clamped to `[0, n_layers]`.
    """
    if half_width is None:
        half_width = max(2, round(0.15 * n_layers))
    mid_start = max(0, trough - half_width)
    mid_end = min(n_layers, trough + half_width)
    high_end = n_layers
    return mid_start, mid_end, high_end


def paper_fixed_schedule(n_layers: int) -> dict[int, float]:
    """The paper's fixed `r1-7b_mgsm.sh` λ schedule, scaled by fraction to `n_layers` (H4 baseline).

    Ablate ~layers 10–19 @+0.3, re-inject ~20–27 @−0.3 on the 28-layer reference; the layer
    fractions are rescaled so the labeled baseline is comparable across the model ladder.
    """
    f = n_layers / _PAPER_REF_LAYERS
    mid_start = round(_PAPER_MID[0] * f)
    mid_end = round(_PAPER_MID[1] * f)
    high_end = n_layers
    return make_schedule(mid_start, mid_end, _PAPER_LAM_MID, high_end, _PAPER_LAM_HIGH, n_layers)


def _boundary_flag(lam_mid, lam_high, mid_choices, high_choices, *, rel_tol: float = 0.0) -> dict:
    """`{"lam_mid": bool, "lam_high": bool}` — True per-λ when the argmax sits at its *dose ceiling*.

    Ceiling = max ablation (`lam_mid` == the max of its grid/range) or max re-injection magnitude
    (`lam_high` == the min of its grid/range). A driver `prog()`s a warning on either, so a
    dose-starved capped arm (e.g. alg1 pinned at its grid ceiling, wanting more ablation than the
    grid allowed) is visible rather than silently reported as the arm's true optimum. `rel_tol`
    (a fraction of the choice span) loosens the equality for continuous optuna ranges; the discrete
    coordinate/grid paths pass the default 0.0 for an exact edge test.
    """
    mid_hi, mid_lo = max(mid_choices), min(mid_choices)
    high_hi, high_lo = max(high_choices), min(high_choices)
    mtol = rel_tol * (mid_hi - mid_lo)
    htol = rel_tol * (high_hi - high_lo)
    return {"lam_mid": bool(lam_mid >= mid_hi - mtol),
            "lam_high": bool(lam_high <= high_lo + htol)}


def _fidelity(df: pd.DataFrame) -> float:
    """Fraction of generative rows whose detected output language matches the item language.

    The E5 §7 fidelity guardrail: a good λ_high keeps accuracy *and* output language, so the search
    doesn't buy accuracy by collapsing output language. NaN when there are no generative rows
    (likelihood-only dev sets have empty `output_lang`).
    """
    if "output_lang" not in df.columns:
        return float("nan")
    det = df[~df["output_lang"].astype(str).isin(["", "?"])]   # exclude likelihood ("") + undetectable
    if len(det) == 0:
        return float("nan")
    return float((det["output_lang"] == det["lang"]).mean())


def _reasoning_fidelity(df: pd.DataFrame) -> float:
    """Fraction of `<think>`-trace rows whose detected trace language matches the item language.

    Zhao `rea_fidelity` — logged alongside the response fidelity so the Fig-3/Fig-4 sweeps carry
    both curves. Present only for reasoning-model rows that emitted a trace (`reasoning_lang` set);
    NaN when the column is absent or empty (all non-reasoning models; every likelihood dev set).
    """
    if "reasoning_lang" not in df.columns:
        return float("nan")
    rl = df["reasoning_lang"]
    rea = df[rl.notna() & ~rl.astype(str).isin(["", "?"])]     # exclude no-trace + undetectable
    if len(rea) == 0:
        return float("nan")
    return float((rea["reasoning_lang"] == rea["lang"]).mean())


def pareto_front(
    df: pd.DataFrame,
    objectives: tuple[str, ...] = ("acc", "fidelity"),
    directions: tuple[str, ...] = ("max", "max"),
) -> pd.Series:
    """Boolean mask of the Pareto-**non-dominated** rows over `objectives` — the accuracy/fidelity
    frontier behind Fig E5.4 (E5 §7).

    A row is *dominated* iff another row is ≥ it on every objective and > it on at least one (after
    flipping any objective whose direction is ``"min"`` so larger is always better). Identical
    objective vectors do not dominate each other, so **ties are all kept**. NaN objective values are
    treated as −∞ (a row with NaN fidelity — e.g. a likelihood-only dev set — can only reach the
    front by leading on the other objective). Works on any search log (coordinate / grid / optuna),
    keeping the multi-objective read-out independent of the sampler. Returns a boolean `Series`
    aligned to `df.index`.
    """
    if len(df) == 0:
        return pd.Series([], dtype=bool, index=df.index)
    signs = [1.0 if d == "max" else -1.0 for d in directions]
    cols = [df[o].to_numpy(dtype=float) * s for o, s in zip(objectives, signs)]
    M = np.where(np.isnan(np.array(cols).T), -np.inf, np.array(cols).T)  # (n, k), higher = better
    n = M.shape[0]
    nondom = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
                nondom[i] = False
                break
    return pd.Series(nondom, index=df.index)


def _is_affine(mapping: dict) -> bool:
    """True when the per-layer mapping holds E6 `AffineEdit`s rather than `(d, r)` subspaces."""
    return bool(mapping) and isinstance(next(iter(mapping.values())), AffineEdit)


def _final_spec(mapping: dict, lam: dict[int, float], scope: str):
    """Freeze the right spec for the mapping type — every search is agnostic to which."""
    if _is_affine(mapping):
        return AffineSpec(edits=mapping, lam=lam, scope=scope)
    return InterventionSpec(subspaces=mapping, lam=lam, scope=scope)


def _eval_schedule_df(eval_fn, model, tok, dev_items, subspaces, lam, *, scope, device,
                      model_name):
    """Freeze the spec, build hooks, eval dev, and return `(spec, per-item df)` — the shared core.

    `_score_schedule` collapses this df to (acc, fidelities); `rank_sweep(keep_items=True)` keeps the
    per-item rows for the pooled 1-SE r\\* selection. Both read the exact same eval, so no path
    double-evaluates.
    """
    spec = _final_spec(subspaces, lam, scope)
    mh = MaskHolder(scope=scope)
    build = build_affine_hooks if isinstance(spec, AffineSpec) else build_layer_hooks
    hooks = build(spec, mask_holder=mh)
    df = eval_fn(model, tok, dev_items, hooks=hooks, mask_holder=mh,
                 device=device, model_name=model_name)
    return spec, df


def _score_schedule(eval_fn, model, tok, dev_items, subspaces, lam, *, scope, device,
                    model_name) -> tuple["InterventionSpec | AffineSpec", float, float, float]:
    spec, df = _eval_schedule_df(eval_fn, model, tok, dev_items, subspaces, lam,
                                 scope=scope, device=device, model_name=model_name)
    acc = float(df["correct"].astype(float).mean())
    return spec, acc, _fidelity(df), _reasoning_fidelity(df)


def coordinate_search(
    model, tok, dev_items, subspaces: dict[int, "object"], n_layers: int, *,
    model_tag: str, grids: dict | None = None, scope: str = "prompt_only",
    device=None, model_name: str | None = None, half_width: int | None = None, eval_fn=None,
    escalate: int = 0, lam_mid_cap: float = 1.0, lam_high_cap: float = -1.0,
) -> tuple[InterventionSpec, pd.DataFrame]:
    """Coordinate/greedy λ search on the **dev** split; returns `(frozen spec*, per-step log)`.

    Two passes (hyperparameter_search.md lever 2): (i) fix bands from this model's ρ trough and
    sweep `λ_mid` ablate-only (`λ_high=0`); (ii) fix the best `λ_mid` and sweep `λ_high`. Each
    candidate is scored by mean dev accuracy via `run_eval`; output-language fidelity is logged
    alongside so accuracy isn't bought by output-language collapse (E5 §7). Touches dev only —
    `subspaces` is built on parallel MATH (E5 §2 data roles).

    `escalate > 0` adds up to that many boundary-triggered widening passes (pass iii): when the
    argmax sits at its dose ceiling (`_boundary_flag`), the grid is extended on exactly the pinned
    side — 1.5× and 2× the pinned edge, clamped to `lam_mid_cap`/`lam_high_cap` — and the sweep
    continues on the new points only. `lam_mid_cap` defaults to 1.0 because a hard projection at
    λ>1 *reflects* (injects the anti-language component) rather than removes more — a different
    treatment, not a bigger dose; gentle-operator callers pass a higher cap. Dev-only and
    deterministic; `attrs["escalation"]` records the passes taken + final grids so a widened-window
    λ* is self-describing in the ledger (E7 execution-flow Q&A Q5).

    `eval_fn` defaults to `evaluate.run_eval` (injectable for tests). `grids` may override
    `{"lam_mid": [...], "lam_high": [...]}`.
    """
    if eval_fn is None:
        from evaluate import run_eval as eval_fn  # local import: keep tuning import-light + stubbable
    grids = grids or {}
    lam_mid_grid = tuple(grids.get("lam_mid", LAM_MID_GRID))
    lam_high_grid = tuple(grids.get("lam_high", LAM_HIGH_GRID))

    trough = trough_layer(model_tag, n_layers)
    mid_start, mid_end, high_end = place_bands(trough, n_layers, half_width=half_width)
    log: list[dict] = []

    def _eval(lam):
        return _score_schedule(eval_fn, model, tok, dev_items, subspaces, lam,
                               scope=scope, device=device, model_name=model_name)

    # (i) ablate-only λ_mid sweep
    best_mid, best_mid_acc = lam_mid_grid[0], float("-inf")
    for lam_mid in lam_mid_grid:
        lam = make_schedule(mid_start, mid_end, lam_mid, high_end, 0.0, n_layers)
        _, acc, fid, rea = _eval(lam)
        log.append({"step": "lam_mid", "lam_mid": lam_mid, "lam_high": 0.0,
                    "acc": acc, "fidelity": fid, "reasoning_fidelity": rea})
        if acc > best_mid_acc:
            best_mid_acc, best_mid = acc, lam_mid

    # (ii) λ_high (re-injection) sweep at the best λ_mid
    best_high, best_high_acc = 0.0, float("-inf")
    for lam_high in lam_high_grid:
        lam = make_schedule(mid_start, mid_end, best_mid, high_end, lam_high, n_layers)
        _, acc, fid, rea = _eval(lam)
        log.append({"step": "lam_high", "lam_mid": best_mid, "lam_high": lam_high,
                    "acc": acc, "fidelity": fid, "reasoning_fidelity": rea})
        if acc > best_high_acc:
            best_high_acc, best_high = acc, lam_high

    # (iii) boundary-triggered one-sided escalation (opt-in; see docstring). Extension points ride
    # the same log with step="lam_mid_esc"/"lam_high_esc"; if an extension moves the λ_mid argmax,
    # λ_high is re-swept at the new λ_mid (those combos are all unseen — pass ii swept the old one).
    n_esc = 0
    boundary = _boundary_flag(best_mid, best_high, lam_mid_grid, lam_high_grid)
    while n_esc < escalate and (boundary["lam_mid"] or boundary["lam_high"]):
        grew = False
        if boundary["lam_mid"]:
            hi = max(lam_mid_grid)
            ext = tuple(dict.fromkeys(round(min(v, lam_mid_cap), 4) for v in (1.5 * hi, 2.0 * hi)))
            ext = tuple(v for v in ext if v > hi)
            if ext:
                grew = True
                for lam_mid in ext:                      # same ablate-only protocol as pass (i)
                    lam = make_schedule(mid_start, mid_end, lam_mid, high_end, 0.0, n_layers)
                    _, acc, fid, rea = _eval(lam)
                    log.append({"step": "lam_mid_esc", "lam_mid": lam_mid, "lam_high": 0.0,
                                "acc": acc, "fidelity": fid, "reasoning_fidelity": rea})
                    if acc > best_mid_acc:
                        best_mid_acc, best_mid = acc, lam_mid
                lam_mid_grid = lam_mid_grid + ext
                if best_mid in ext:                      # argmax moved -> λ_high re-sweep at new λ_mid
                    best_high, best_high_acc = 0.0, float("-inf")
                    for lam_high in lam_high_grid:
                        lam = make_schedule(mid_start, mid_end, best_mid, high_end, lam_high, n_layers)
                        _, acc, fid, rea = _eval(lam)
                        log.append({"step": "lam_high_esc", "lam_mid": best_mid, "lam_high": lam_high,
                                    "acc": acc, "fidelity": fid, "reasoning_fidelity": rea})
                        if acc > best_high_acc:
                            best_high_acc, best_high = acc, lam_high
        if boundary["lam_high"]:
            lo = min(lam_high_grid)
            ext = tuple(dict.fromkeys(round(max(v, lam_high_cap), 4) for v in (1.5 * lo, 2.0 * lo)))
            ext = tuple(v for v in ext if v < lo)
            if ext:
                grew = True
                for lam_high in ext:
                    lam = make_schedule(mid_start, mid_end, best_mid, high_end, lam_high, n_layers)
                    _, acc, fid, rea = _eval(lam)
                    log.append({"step": "lam_high_esc", "lam_mid": best_mid, "lam_high": lam_high,
                                "acc": acc, "fidelity": fid, "reasoning_fidelity": rea})
                    if acc > best_high_acc:
                        best_high_acc, best_high = acc, lam_high
                lam_high_grid = lam_high_grid + ext
        if not grew:
            break                                        # pinned side(s) already at the physical cap
        n_esc += 1
        boundary = _boundary_flag(best_mid, best_high, lam_mid_grid, lam_high_grid)

    final_lam = make_schedule(mid_start, mid_end, best_mid, high_end, best_high, n_layers)
    spec = _final_spec(subspaces, final_lam, scope)
    log_df = pd.DataFrame(log)
    log_df.attrs["bands"] = {"trough": trough, "mid_start": mid_start, "mid_end": mid_end,
                             "high_end": high_end, "lam_mid": best_mid, "lam_high": best_high}
    log_df.attrs["boundary"] = boundary
    log_df.attrs["escalation"] = {"passes": n_esc, "lam_mid_grid": list(lam_mid_grid),
                                  "lam_high_grid": list(lam_high_grid)}
    return spec, log_df


def grid_search(
    model, tok, dev_items, subspaces: dict[int, "object"], n_layers: int, *,
    model_tag: str, grids: dict | None = None, scope: str = "prompt_only",
    device=None, model_name: str | None = None, half_width: int | None = None, eval_fn=None,
) -> tuple[InterventionSpec, pd.DataFrame]:
    """Joint `λ_mid × λ_high` grid search on the **dev** split; returns `(best spec, grid log)`.

    The *ceteris-paribus* companion to `coordinate_search`: bands are held fixed (seeded from this
    model's E3 ρ trough, exactly as there), and the **full** `LAM_MID_GRID × LAM_HIGH_GRID` product
    is scored jointly — so a `λ_mid`/`λ_high` interaction that greedy coordinate descent would miss
    (it sweeps them one at a time) is captured. Same seams as `coordinate_search` (`_score_schedule`
    → `eval_fn`, default `evaluate.run_eval`), same dev-only data role (E5 §2); mean dev accuracy is
    the argmax objective, output-language fidelity is logged alongside (E5 §7). `attrs["bands"]`
    records the frozen band placement; `grids` may override `{"lam_mid": [...], "lam_high": [...]}`.
    """
    if eval_fn is None:
        from evaluate import run_eval as eval_fn  # local import: keep tuning import-light + stubbable
    grids = grids or {}
    lam_mid_grid = tuple(grids.get("lam_mid", LAM_MID_GRID))
    lam_high_grid = tuple(grids.get("lam_high", LAM_HIGH_GRID))

    trough = trough_layer(model_tag, n_layers)
    mid_start, mid_end, high_end = place_bands(trough, n_layers, half_width=half_width)
    log: list[dict] = []
    best_spec, best_acc = None, float("-inf")
    best_mid, best_high = lam_mid_grid[0], lam_high_grid[0]
    for lam_mid in lam_mid_grid:
        for lam_high in lam_high_grid:
            lam = make_schedule(mid_start, mid_end, lam_mid, high_end, lam_high, n_layers)
            spec, acc, fid, rea = _score_schedule(eval_fn, model, tok, dev_items, subspaces, lam,
                                                  scope=scope, device=device, model_name=model_name)
            log.append({"lam_mid": lam_mid, "lam_high": lam_high, "acc": acc, "fidelity": fid,
                        "reasoning_fidelity": rea})
            if acc > best_acc:
                best_acc, best_spec, best_mid, best_high = acc, spec, lam_mid, lam_high
    log_df = pd.DataFrame(log)
    log_df.attrs["bands"] = {"trough": trough, "mid_start": mid_start, "mid_end": mid_end,
                             "high_end": high_end}
    log_df.attrs["boundary"] = _boundary_flag(best_mid, best_high, lam_mid_grid, lam_high_grid)
    return best_spec, log_df


def rank_sweep(
    model, tok, dev_items, build_fn, lam: dict[int, float], n_layers: int, *,
    ranks=RANK_GRID, scope: str = "prompt_only", device=None,
    model_name: str | None = None, eval_fn=None, dose_fn=None, keep_items: bool = False,
) -> tuple[int, pd.DataFrame]:
    """Accuracy-vs-rank curve on the **dev** split at a frozen λ schedule; returns
    `(best_rank, curve_df)`.

    Lever 1 / H5 (hyperparameter_search.md): the λ schedule is layer-based and subspace-agnostic,
    so a single frozen `lam` (the r=10 λ\\*) is reused across every rank and the sweep is a clean
    `argmax_dev acc(rank | λ*)`. `build_fn(rank) -> {layer: M_s}` is injected (the notebook passes a
    `build_layer_subspaces(...)` closure), keeping this import-light and stubbable; `eval_fn`
    defaults to `evaluate.run_eval`.

    For each `r`: score `InterventionSpec(build_fn(r), lam, scope)` on dev via the same
    `_score_schedule` path as `coordinate_search`, logging `{rank, acc, fidelity}`. The **full**
    curve is always returned (never collapsed to just `best_rank`), per the E5 §7 rank-parity
    control.

    `keep_items=True` additionally stashes each rank's per-item `{"ids": [...], "correct": [...]}`
    into ``curve_df.attrs["items"]`` (keyed by rank) — the raw material the pooled 1-SE r\\* selection
    (`select_rank_1se`) needs, since a mean-accuracy curve alone can't produce a cluster-bootstrap SE.
    The default path (`keep_items=False`) is byte-identical to before.
    """
    if eval_fn is None:
        from evaluate import run_eval as eval_fn  # local import: keep tuning import-light + stubbable
    curve: list[dict] = []
    items: dict[int, dict] = {}
    best_rank, best_acc = None, float("-inf")
    for r in ranks:
        subspaces = build_fn(r)
        if keep_items:                                     # keep the per-item df for the 1-SE selection
            _, df = _eval_schedule_df(eval_fn, model, tok, dev_items, subspaces, lam,
                                      scope=scope, device=device, model_name=model_name)
            acc = float(df["correct"].astype(float).mean())
            fid, rea = _fidelity(df), _reasoning_fidelity(df)
            ids = df["id"].tolist() if "id" in df.columns else list(range(len(df)))
            items[int(r)] = {"ids": ids, "correct": df["correct"].astype(int).tolist()}
        else:
            _, acc, fid, rea = _score_schedule(eval_fn, model, tok, dev_items, subspaces, lam,
                                               scope=scope, device=device, model_name=model_name)
        row = {"rank": int(r), "acc": acc, "fidelity": fid, "reasoning_fidelity": rea}
        if dose_fn is not None:                            # H3 de-confound: dose grows with rank
            row["dose"] = float(dose_fn(subspaces))        # Σλ²e_l at this rank's projection
        curve.append(row)
        if acc > best_acc:
            best_acc, best_rank = acc, int(r)
    curve_df = pd.DataFrame(curve)
    if keep_items:
        curve_df.attrs["items"] = items
    return best_rank, curve_df


def rank_sweep_retuned(
    model, tok, dev_items, test_items, build_fn, n_layers: int, *,
    tune_fn, ranks=RANK_GRID, device=None, model_name: str | None = None,
    eval_fn=None, dose_fn=None, keep_items: bool = False,
) -> tuple[int, pd.DataFrame]:
    """Accuracy-vs-rank curve with λ **re-tuned per rank** (equal footing) — dev-tune, test-score.

    The E7 unification of `rank_sweep` (one frozen λ across ranks) and the old below-10 sweep
    (per-rank λ, low-resource only, now folded into nb08): every rank ``r`` gets its OWN dev-tuned λ\\* via
    ``tune_fn(build_fn(r), dev_items) -> (spec, log)``; the frozen spec is then scored on the held-out
    ``test_items``. λ is an OUTPUT per rank, logged in the ``lam`` column — so a rank below or above
    r=10 competes at its own best schedule, not r=10's. Returns ``(argmax-test-acc rank [descriptive
    only], curve_df)`` with columns ``rank, lam, acc, fidelity, reasoning_fidelity`` (+ ``dose`` when
    ``dose_fn`` is given).

    - ``tune_fn(subs, dev_items) -> (spec, log_df)`` — injected (the notebook closes it over its
      ``_search`` + ``common`` dict), keeping this file decoupled from ``E5_SEARCH``/``MODEL_TAG``.
    - ``dose_fn`` — the *factory* ``lam -> (subs -> float)`` (nb08's ``_dose_fn`` passed **un-applied**),
      so dose is read at each rank's OWN retuned λ (unlike `rank_sweep`, where it takes the applied
      ``subs -> float`` at the single frozen λ).
    - ``keep_items=True`` stashes each rank's **dev** per-item correctness at that rank's retuned λ into
      ``curve_df.attrs["items"]`` (keyed by rank) — the substrate for `select_rank_1se`. **Invariant:**
      curve rows are TEST metrics; ``attrs["items"]`` is DEV — selection reads dev, the reported curve
      reads test, so no test peeking (the tune-on-dev / score-on-test split of the old below-10 sweep).

    ``dev_items``/``test_items`` are each passed pre-subsampled once and reused across every rank, so
    the paired discipline (same dev subsample, same test items across ranks) holds by construction.
    """
    if eval_fn is None:
        from evaluate import run_eval as eval_fn  # local import: keep tuning import-light + stubbable
    curve: list[dict] = []
    items: dict[int, dict] = {}
    best_rank, best_acc = None, float("-inf")
    for r in ranks:
        subs = build_fn(r)                                 # DiffPCA M_s at rank r (language-agnostic)
        spec, _log = tune_fn(subs, dev_items)              # λ* search on DEV ONLY (== coordinate_search)
        # score the retuned spec on the HELD-OUT test split (test never entered the λ search)
        _, test_df = _eval_schedule_df(eval_fn, model, tok, test_items, subs, spec.lam,
                                       scope=spec.scope, device=device, model_name=model_name)
        acc = float(test_df["correct"].astype(float).mean())
        row = {"rank": int(r), "lam": str(spec.lam), "acc": acc,
               "fidelity": _fidelity(test_df), "reasoning_fidelity": _reasoning_fidelity(test_df)}
        if dose_fn is not None:                            # dose at THIS rank's retuned λ (H3 de-confound)
            row["dose"] = float(dose_fn(spec.lam)(subs))
        if keep_items:                                     # DEV per-item @ retuned λ -> 1-SE substrate
            _, dev_df = _eval_schedule_df(eval_fn, model, tok, dev_items, subs, spec.lam,
                                          scope=spec.scope, device=device, model_name=model_name)
            ids = dev_df["id"].tolist() if "id" in dev_df.columns else list(range(len(dev_df)))
            items[int(r)] = {"ids": ids, "correct": dev_df["correct"].astype(int).tolist()}
        curve.append(row)
        if acc > best_acc:
            best_acc, best_rank = acc, int(r)
    curve_df = pd.DataFrame(curve)
    if keep_items:
        curve_df.attrs["items"] = items
    return best_rank, curve_df


def layer_sweep(
    model, tok, dev_items, subspaces: dict[int, "object"], n_layers: int, *,
    lam_mid: float, lam_high: float = 0.0, band_width: int | None = None,
    starts=None, step: int | None = None, scope: str = "prompt_only", device=None,
    model_name: str | None = None, eval_fn=None,
) -> tuple[int, pd.DataFrame]:
    """Accuracy(+fidelity)-vs-starting-layer curve at a frozen λ — the Zhao Fig-4 axis.

    Mirrors `rank_sweep`, but sweeps the ablate band's START layer across depth at a fixed band
    width and frozen `(lam_mid, lam_high)`; the re-inject band fills from the band end to the top.
    `subspaces` is built once (layer-keyed) and reused, so this is a clean `argmax_dev acc(start|λ*)`.
    The **full** curve is returned (never collapsed), logging `acc`, response `fidelity`, and
    `reasoning_fidelity` per start (the two fidelity curves Fig 4 plots). `build_fn` is not needed —
    the subspace is layer-based, so only the λ band placement moves.
    """
    if eval_fn is None:
        from evaluate import run_eval as eval_fn  # local import: keep tuning import-light + stubbable
    width = band_width if band_width is not None else max(1, round(0.3 * n_layers))
    if starts is None:
        stp = step if step is not None else max(1, round(0.1 * n_layers))
        starts = list(range(0, max(1, n_layers - width + 1), stp))
    curve: list[dict] = []
    best_start, best_acc = None, float("-inf")
    for s in starts:
        lam = make_schedule(int(s), min(int(s) + width, n_layers), lam_mid, n_layers, lam_high, n_layers)
        _, acc, fid, rea = _score_schedule(eval_fn, model, tok, dev_items, subspaces, lam,
                                           scope=scope, device=device, model_name=model_name)
        curve.append({"start": int(s), "acc": acc, "fidelity": fid, "reasoning_fidelity": rea})
        if acc > best_acc:
            best_acc, best_start = acc, int(s)
    return best_start, pd.DataFrame(curve)


def strength_sweep(
    model, tok, dev_items, subspaces: dict[int, "object"], n_layers: int, *,
    trough: int | None = None, model_tag: str | None = None,
    strengths=LAM_SWEEP_SYMMETRIC, band: tuple[int, int] | None = None,
    scope: str = "prompt_only", device=None, model_name: str | None = None, eval_fn=None,
) -> pd.DataFrame:
    """Single-band signed-λ sweep for the Zhao Fig-3 axis (+ removes / − re-injects language).

    One ablate band (centered on the ρ trough via `place_bands`, or an explicit `band=(start, end)`)
    swept across `strengths` (default −0.4..+0.4; strength 0 => empty schedule => clean), with no
    separate re-inject band — the exact Fig-3 x-axis, rather than stitching the coordinate search's
    asymmetric two-band grid. Logs `acc`, response `fidelity`, and `reasoning_fidelity` per strength.
    Returns the full log DataFrame (the dose frontier / Fig-3 clone reads straight off it).
    """
    if eval_fn is None:
        from evaluate import run_eval as eval_fn  # local import: keep tuning import-light + stubbable
    if band is None:
        t = trough if trough is not None else trough_layer(model_tag, n_layers)
        mid_start, mid_end, _ = place_bands(t, n_layers)
    else:
        mid_start, mid_end = band
    log: list[dict] = []
    for lam in strengths:
        sched = make_schedule(mid_start, mid_end, float(lam), n_layers, 0.0, n_layers)
        _, acc, fid, rea = _score_schedule(eval_fn, model, tok, dev_items, subspaces, sched,
                                           scope=scope, device=device, model_name=model_name)
        log.append({"strength": float(lam), "acc": acc, "fidelity": fid, "reasoning_fidelity": rea})
    return pd.DataFrame(log)


def lam_from_params(params: dict, n_layers: int, per_layer: bool) -> dict[int, float]:
    """Rebuild a `{layer -> λ}` schedule from an Optuna trial's sampled params.

    Band mode reads the two independent bands (`make_bands`); `per_layer` mode reads a free λ per
    layer. Shared by the objective and the post-hoc Pareto-spec reconstruction so both see the exact
    same schedule. **Public** so both drivers can reconstruct each optuna row's schedule off the
    searchlog `params_*` columns for the search-agnostic dose annotation (Group B).
    """
    if per_layer:
        return {l: params[f"lam_{l}"] for l in range(n_layers)
                if abs(params[f"lam_{l}"]) > 1e-9}
    return make_bands(
        ablate=(params["a_start"], params["a_end"], params["lam_mid"]),
        inject=(params["i_start"], params["i_end"], params["lam_high"]),
        n_layers=n_layers,
    )


_lam_from_params = lam_from_params   # backward-compat alias (pre-promotion private name)


def optuna_search(
    model, tok, dev_items, subspaces: dict[int, "object"], n_layers: int, *,
    model_tag: str, n_trials: int = 40, sampler=None, per_layer: bool = False,
    lam_mid_range: tuple[float, float] = (0.1, 0.5),
    lam_high_range: tuple[float, float] = (-0.3, 0.0),
    per_layer_range: tuple[float, float] = (-0.5, 0.5),
    scope: str = "prompt_only", device=None, model_name: str | None = None,
    seed: int = 0, eval_fn=None,
    escalate: int = 0, lam_mid_cap: float = 1.0, lam_high_cap: float = -1.0,
) -> tuple["object", list[InterventionSpec]]:
    """Multi-objective search that **searches the layers**, not just the λs (E5 lever 2, un-frozen).

    Band mode (default) makes the ablate band `[a_start, a_end)` and the re-inject band
    `[i_start, i_end)` genuine search variables (ordered `a_start < a_end ≤ i_start < i_end`, so a gap
    of untouched layers is allowed) alongside `λ_mid ∈ lam_mid_range` and `λ_high ∈ lam_high_range` —
    so the search can relocate the ablate/inject layers off the E3 ρ trough instead of taking geometry
    for granted. The trough-centered `place_bands` schedule (+ paper-adjacent λ) is **enqueued as the
    first trial**, so the geometry prior is a warm start the search may leave, not a fixed point.

    Optimizes **two objectives jointly** — mean dev accuracy and output-language fidelity (E5 §7) — and
    returns the Pareto-optimal `InterventionSpec`s (`study.best_trials`), exactly the data behind
    Fig E5.4. `sampler` is injectable (any `optuna.samplers.BaseSampler`); default is multivariate TPE
    (sample-efficient at E5's small dev-eval budget). `per_layer=True` searches a free λ per layer
    instead (dimension ≈ `n_layers`) — the reparameterization-audit flavour, reachable by passing e.g.
    a `CmaEsSampler`. Dev-only (E5 §2); `eval_fn` defaults to `evaluate.run_eval`.

    `escalate > 0` (band mode only) adds up to that many boundary-triggered widening passes: when
    the frozen accuracy-champion sits at a λ range edge (`_boundary_flag`, rel_tol=1e-3), the range
    is widened on exactly the pinned side by one span (clamped to `lam_mid_cap`/`lam_high_cap` —
    the projection family's mid cap is 1.0, the reflection wall; see `coordinate_search`), a fresh
    study warm-starts from the pinned champion, and the prior pass's trials are merged back in
    (`add_trials`) so `trials_dataframe`/`best_trials` span every pass. `user_attrs["escalation"]`
    records the passes taken + final ranges (E7 execution-flow Q&A Q5).
    """
    import optuna  # core dep, but imported lazily so the coordinate/grid path stays import-light
    if eval_fn is None:
        from evaluate import run_eval as eval_fn  # local import: keep tuning import-light + stubbable
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _default_sampler = sampler is None
    if _default_sampler:
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    # working λ windows — rebound by the escalation loop; `objective` reads the current binding
    mid_range, high_range = tuple(lam_mid_range), tuple(lam_high_range)

    def objective(trial):
        if per_layer:
            p = {f"lam_{l}": trial.suggest_float(f"lam_{l}", *per_layer_range)
                 for l in range(n_layers)}
        else:
            a_start = trial.suggest_int("a_start", 0, n_layers - 2)
            a_end = trial.suggest_int("a_end", a_start + 1, n_layers - 1)
            i_start = trial.suggest_int("i_start", a_end, n_layers - 1)
            i_end = trial.suggest_int("i_end", i_start + 1, n_layers)
            p = {"a_start": a_start, "a_end": a_end, "i_start": i_start, "i_end": i_end,
                 "lam_mid": trial.suggest_float("lam_mid", *mid_range),
                 "lam_high": trial.suggest_float("lam_high", *high_range)}
        lam = lam_from_params(p, n_layers, per_layer)
        _, acc, fid, _rea = _score_schedule(eval_fn, model, tok, dev_items, subspaces, lam,
                                            scope=scope, device=device, model_name=model_name)
        return acc, (0.0 if np.isnan(fid) else fid)

    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
    if not per_layer:
        mid_start, mid_end, high_end = place_bands(trough_layer(model_tag, n_layers), n_layers)
        # clamp the geometry seed into the search ranges so the enqueued warm-start trial is valid
        a_start = min(max(0, mid_start), n_layers - 2)
        a_end = min(max(a_start + 1, mid_end), n_layers - 1)
        i_start = min(max(a_end, mid_end), n_layers - 1)
        i_end = min(max(i_start + 1, high_end), n_layers)
        lam_mid0 = min(max(mid_range[0], 0.4), mid_range[1])
        lam_high0 = min(max(high_range[0], -0.3), high_range[1])
        study.enqueue_trial({"a_start": a_start, "a_end": a_end, "i_start": i_start,
                             "i_end": i_end, "lam_mid": lam_mid0, "lam_high": lam_high0})
    study.optimize(objective, n_trials=n_trials)

    # Capture the geometry warm-start NOW: after an escalation pass the merged study's trials[0] is
    # the escalation warm-start (the pass-1 champion), not the ρ-trough seed — the Group-G
    # warm-start provenance must keep pointing at the true seed (optuna_warmstart_delta prefers
    # this attr over trials[0]).
    _geom_seed = None
    if not per_layer and study.trials:
        _t0 = study.trials[0]
        _geom_seed = {"params": dict(_t0.params), "values": [float(v) for v in (_t0.values or [])]}

    # Boundary-triggered one-sided window escalation (opt-in; see docstring). Each pass widens only
    # the pinned side by one span (clamped to its cap), re-searches with the pinned champion
    # enqueued, then merges the prior trials back so the log + Pareto set are cumulative.
    n_esc = 0
    while not per_layer and n_esc < escalate and study.best_trials:
        champ = max(study.best_trials, key=lambda t: t.values[0])
        b = _boundary_flag(champ.params["lam_mid"], champ.params["lam_high"],
                           mid_range, high_range, rel_tol=1e-3)
        new_mid = ((mid_range[0], round(min(lam_mid_cap, mid_range[1] + (mid_range[1] - mid_range[0])), 4))
                   if b["lam_mid"] else mid_range)
        new_high = ((round(max(lam_high_cap, high_range[0] - (high_range[1] - high_range[0])), 4), high_range[1])
                    if b["lam_high"] else high_range)
        if (new_mid, new_high) == (mid_range, high_range):
            break                                       # not pinned, or pinned side already at its cap
        mid_range, high_range = new_mid, new_high
        prev = study
        esc_sampler = (optuna.samplers.TPESampler(multivariate=True, seed=seed + 100 + n_esc)
                       if _default_sampler else sampler)
        study = optuna.create_study(directions=["maximize", "maximize"], sampler=esc_sampler)
        study.enqueue_trial(dict(champ.params))         # warm start: the pinned champion
        study.optimize(objective, n_trials=n_trials)
        study.add_trials(prev.get_trials(deepcopy=False))
        n_esc += 1

    if _geom_seed is not None:
        study.set_user_attr("geometry_seed", _geom_seed)

    # Boundary-argmax flag for the accuracy-champion trial (the spec a driver freezes): stored on the
    # study so the driver — which rebuilds its own searchlog from `study.trials_dataframe` — can
    # surface it. Band mode only (a per-layer search has no single λ_mid/λ_high to sit at a range edge).
    if not per_layer and study.best_trials:
        champ = max(study.best_trials, key=lambda t: t.values[0])
        study.set_user_attr("boundary", _boundary_flag(
            champ.params["lam_mid"], champ.params["lam_high"],
            mid_range, high_range, rel_tol=1e-3))
        study.set_user_attr("escalation", {"passes": n_esc, "lam_mid_range": list(mid_range),
                                           "lam_high_range": list(high_range)})

    specs = [_final_spec(subspaces, lam_from_params(t.params, n_layers, per_layer), scope)
             for t in study.best_trials]
    return study, specs


def optuna_warmstart_delta(study, chosen_trial) -> dict:
    """Provenance: compare the enqueued geometry warm-start (trial 0) to the `chosen_trial` kept.

    `optuna_search` enqueues the ρ-trough `place_bands` seed as the first trial and (band mode)
    stashes it as `user_attrs["geometry_seed"]` — preferred here over `study.trials[0]`, because
    after a boundary-escalation pass the merged study's first trial is the escalation warm-start
    (the pass-1 champion), not the geometry seed. `chosen_trial` is what the search actually
    selected. Returns a flat provenance dict — the warm/chosen ablate+inject band edges, the signed
    ablate-band-center shift (`band_shift`), whether the search departed the seed band at all
    (`left_trough`), and the two accuracy objectives (`warm_acc`, `chosen_acc`) so a driver can log
    `chosen_acc − warm_acc` ("did leaving the geometry seed buy accuracy"). Band mode only —
    returns `{}` for a per-layer study, which has no band edges to compare. This makes "did the
    geometry warm-start survive the search" a reportable artifact.
    """
    _ga = getattr(study, "user_attrs", {}).get("geometry_seed")
    if _ga:
        warm_params, warm_values = _ga["params"], list(_ga.get("values") or [])
    else:                                                  # pre-escalation studies: trial 0 IS the seed
        warm_params, warm_values = study.trials[0].params, list(study.trials[0].values or [])
    edges = ("a_start", "a_end", "i_start", "i_end")
    if any(e not in warm_params for e in edges) or any(e not in chosen_trial.params for e in edges):
        return {}                                          # per-layer search: no band edges
    def _center(p):
        return 0.5 * (p["a_start"] + p["a_end"])
    moved = any(warm_params[e] != chosen_trial.params[e] for e in edges)
    return {
        "warm_a_start": int(warm_params["a_start"]), "warm_a_end": int(warm_params["a_end"]),
        "warm_i_start": int(warm_params["i_start"]), "warm_i_end": int(warm_params["i_end"]),
        "chosen_a_start": int(chosen_trial.params["a_start"]),
        "chosen_a_end": int(chosen_trial.params["a_end"]),
        "chosen_i_start": int(chosen_trial.params["i_start"]),
        "chosen_i_end": int(chosen_trial.params["i_end"]),
        "band_shift": float(_center(chosen_trial.params) - _center(warm_params)),
        "left_trough": bool(moved),
        "warm_acc": float(warm_values[0]) if warm_values else float("nan"),
        "chosen_acc": float(chosen_trial.values[0]) if chosen_trial.values else float("nan"),
    }


def select_rank_1se(pooled_correct_by_rank, cluster_by_rank, *, n_boot: int = 1000, seed: int = 0):
    """1-SE parsimony rank selection on pooled dev items (the r\\* headline; H5).

    `pooled_correct_by_rank[r]` is the 0/1 per-item correctness vector at rank `r` (pooled across the
    low-resource dev langs); `cluster_by_rank[r]` the matching problem-cluster labels (so a problem's
    parallel MGSM rows resample together). `r_argmax` maximizes pooled dev accuracy; `se` is the
    **cluster-bootstrap standard error** of the accuracy at `r_argmax`; and **r\\* is the smallest rank
    whose accuracy is within one `se` of the max** (Breiman's 1-SE rule) — so r\\* stops chasing dev
    noise past the point where a smaller subspace is statistically indistinguishable. Returns
    `(r_star, r_argmax, se)`.
    """
    ranks = sorted(pooled_correct_by_rank)
    accs = {r: float(np.mean(pooled_correct_by_rank[r])) for r in ranks}
    r_argmax = max(ranks, key=lambda r: accs[r])
    x = np.asarray(pooled_correct_by_rank[r_argmax], dtype=np.float64).ravel()
    c = np.asarray(cluster_by_rank[r_argmax]).ravel()
    _, inv = np.unique(c, return_inverse=True)             # same cluster resample as geometry.py
    K = int(inv.max()) + 1
    members = [np.where(inv == k)[0] for k in range(K)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, K, K)
        boots[b] = x[np.concatenate([members[k] for k in pick])].mean()
    se = float(np.std(boots, ddof=1))
    thresh = accs[r_argmax] - se
    r_star = min(r for r in ranks if accs[r] >= thresh)    # smallest rank within 1 SE of the peak
    return int(r_star), int(r_argmax), se
