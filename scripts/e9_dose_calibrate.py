#!/usr/bin/env python
"""E9 dose calibration: solve the per-language lambda grid off the activation cache. CPU, no GPU.

Prerequisite for `scripts/e9_tradeoff_grid.py`. Spec: docs/workshop/tradeoff_grid.md; the design
argument: docs/workshop/design_rationale.md "Why lambda is dose-calibrated, and per language".

WHAT THIS SOLVES, and why it needs no generation. A schedule's dose is `sum_l lam_l^2 e_l` (`src/
dose.py`), and `e_l` -- the unit-lambda edit energy fraction -- is measured on cached activations.
Dose is quadratic in a uniform rescale, so hitting a dose target is closed form, not a search.

THE HYBRID ANCHOR (the settled reading of the two docs, which disagreed):

  * The **centre rung** is Zhao's published global lambda VERBATIM in every language (`ablate the
    mid band @ +0.3, re-inject the high band @ -0.3`, depth-scaled by `tuning.paper_fixed_schedule`).
    So the grid's anchor stays a number we did not choose, exactly as tradeoff_grid.md:20 claims.
    Its dose still differs per language -- that is the fact motivating everything below.
  * The **off-centre rungs (0.5x, 1.5x, 2.0x)** target one ABSOLUTE dose per rung, the rung
    multiple of the cross-language anchor dose (geometric mean over the 11 languages of the centre
    dose at rank L-1). So `lambda` genuinely varies per language there, and a rung means the same
    SIZE of edit in bn as in de -- design_rationale.md:22-23's claim, which a per-language-relative
    solve could not deliver (it collapses to alpha = sqrt(k), i.e. one lambda everywhere).

  Consequence to state in the caption, not bury: the centre deliberately sits off its own rungs'
  absolute dose scale. You cannot have both an untouched published anchor AND cross-language dose
  matching at the same point; we keep the anchor at the centre and match doses around it.

COST. The heavy work is `d x d` covariances and eigendecompositions -- at d=3584 the dominant term.
Two caches make the sweep affordable: `operators.layer_whitener` computes each layer's rank- and
basis-independent second moments once (19 cells would otherwise redo them), and the diffPCA span is
built once per layer at the maximum rank and sliced (PCA nests exactly; alg1 does NOT -- its
two-step SVD refines a rank-dependent anchor -- so alg1 spans are rebuilt per rank, which is cheap,
being an SVD of the (L, d) mean matrix).

OUTPUTS
  results/E9/<tag>/calib_<tag>.csv          one row per (basis, operator, rank, rung, lang):
                                            lambda_mid, lambda_high, dose_target, dose_measured,
                                            dose_centre, alpha, clamped, lambda_source, ...
  results/E9/<tag>/specs/<cell>.pt          frozen per-layer AffineEdits per (basis, operator, rank)
  results/E9/<tag>/E9_calib_progress_<tag>.log

KNOBS
  E5_MODEL=Qwen/Qwen2.5-7B-Instruct   model (tag derives from it)
  E5_N_SAMPLES=7500                   construction-corpus size == which cache to read
  E9_DOSE_ROWS=512                    rows per language used to estimate e_l (None/0 = all)
  E9_DOSE_POOL=geo                    geo|mean -- how per-language doses pool into the anchor
  E9_SHRINK=0.1                        covariance ridge for leace/wiener
  E9_LANGS=                            comma list; default = every language in the cache
  E9_DRYRUN=1                          print the plan (cells, layers, arms) and exit
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
             if (p / "src" / "config.py").exists())
sys.path.insert(0, str(_root / "src"))

import pandas as pd
import torch

from candidates import build_layer_subspaces, layer_slice
from config import ACTS_DIR, Config, results_path, set_seeds
from dose import edit_energy_by_lang, pool_dose, scale_schedule, schedule_dose, solve_alpha
from e9_grid import CENTRE_RUNG, centre_rank, grid_arms, grid_cells
from extract import load_or_extract_acts
from models import angular_forced
from operators import build_family_edits, layer_whitener
from tuning import paper_fixed_schedule

set_seeds(0)

cfg = Config(model_name=os.environ.get("E5_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
             n_samples=int(os.environ.get("E5_N_SAMPLES", "7500")))
TAG = cfg.model_name.split("/")[-1]
_rows = int(os.environ.get("E9_DOSE_ROWS", "512"))
DOSE_ROWS = None if _rows <= 0 else _rows
POOL = os.environ.get("E9_DOSE_POOL", "geo")
SHRINK = float(os.environ.get("E9_SHRINK", "0.1"))
DRYRUN = os.environ.get("E9_DRYRUN", "") not in ("", "0")

# results_path routes on the `E9_` filename prefix (config._EXP_RE), so every artifact below
# carries it and lands in results/E9/<tag>/ — there is no `experiment=` kwarg.
CALIB_CSV = results_path(f"E9_calib_{TAG}.csv", TAG)
PROG = results_path(f"E9_calib_progress_{TAG}.log", TAG)
SPEC_DIR = CALIB_CSV.parent / "e9_specs"


def prog(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    PROG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def cell_tag(basis: str, operator: str, rank: int) -> str:
    return f"{basis}_{operator}_r{int(rank)}"


# --- the cache ------------------------------------------------------------------------------------
cache_pt = ACTS_DIR / f"acts_{TAG}_n{cfg.n_samples}_stratified-s0.pt"
if not cache_pt.exists():
    sys.exit(f"\nMISSING n={cfg.n_samples} cache: {cache_pt}\n"
             f"Refusing to run — load_or_extract_acts would RE-EXTRACT (hours on GPU).\n"
             f"On Prajna run the 'extract' phase first:\n"
             f"  sbatch scripts/prajna/e9_prajna.sbatch extract {cfg.model_name}\n")

prog(f"=== E9 dose calibration | {TAG} | n={cfg.n_samples} | shrink={SHRINK} "
     f"| dose_rows={DOSE_ROWS or 'all'} | pool={POOL} ===")
prog(f"loading {cache_pt.name} ({cache_pt.stat().st_size / 2**30:.1f} GiB) ...")
acts = load_or_extract_acts(cfg)
LANGS = (os.environ["E9_LANGS"].split(",") if os.environ.get("E9_LANGS") else list(acts))
missing = [l for l in LANGS if l not in acts]
if missing:
    sys.exit(f"cache has no language(s) {missing} (have {sorted(acts)})")
N_LAYERS = int(next(iter(acts.values())).shape[1])
N_LANGS = len(acts)                      # the BASIS is built from every cached language, always
D_MODEL = int(next(iter(acts.values())).shape[2])

# --- Zhao's schedule: the bands are fixed in every arm, only the strength moves -------------------
LAM_ZHAO = paper_fixed_schedule(N_LAYERS)
EDIT_LAYERS = sorted(LAM_ZHAO)
_abl = [l for l, v in LAM_ZHAO.items() if v > 0]
_inj = [l for l, v in LAM_ZHAO.items() if v < 0]
prog(f"cache: {N_LANGS} langs, {N_LAYERS} layers, d={D_MODEL}; eval langs {LANGS}")
prog(f"paper_fixed_schedule({N_LAYERS}): ablate {min(_abl)}..{max(_abl)} @ +{LAM_ZHAO[min(_abl)]} | "
     f"re-inject {min(_inj)}..{max(_inj)} @ {LAM_ZHAO[min(_inj)]}  ({len(EDIT_LAYERS)} edited layers)")

ARMS = grid_arms(N_LANGS)
CELLS = grid_cells(ARMS)
R0 = centre_rank(N_LANGS)
DIFFPCA_ANGULAR = True                   # the shipped default; forced on massive-activation models
prog(f"grid: {len(ARMS)} arms over {len(CELLS)} distinct (basis, operator, rank) edit-sets; "
     f"centre rank {R0}; diffpca angular={DIFFPCA_ANGULAR} "
     f"(angular_forced({TAG})={angular_forced(TAG)})")
for stage in ("centre", "lambda", "rank"):
    names = [a.name for a in ARMS if a.stage == stage]
    prog(f"  stage {stage:7s} ({len(names):2d}): {', '.join(names)}")

if DRYRUN:
    prog("E9_DRYRUN set — plan printed, nothing computed")
    sys.exit(0)

# --- pass 1: the spans. diffPCA nests (build once at max rank, slice); alg1 does not. -------------
RANKS_BY_BASIS: dict[str, list[int]] = {}
for basis, _op, rank in CELLS:
    RANKS_BY_BASIS.setdefault(basis, [])
    if rank not in RANKS_BY_BASIS[basis]:
        RANKS_BY_BASIS[basis].append(rank)
for b in RANKS_BY_BASIS:
    RANKS_BY_BASIS[b].sort()
prog(f"spans needed: " + "; ".join(f"{b} r={rs}" for b, rs in RANKS_BY_BASIS.items()))

SPANS: dict[tuple[str, int], dict[int, torch.Tensor]] = {}
t0 = time.time()
for basis, ranks in RANKS_BY_BASIS.items():
    if basis == "diffpca":
        rmax = max(ranks)
        big = build_layer_subspaces(acts, EDIT_LAYERS, name="diffpca", rank=rmax,
                                    diffpca_angular=DIFFPCA_ANGULAR)
        for r in ranks:                  # PCA prefix: verified nested, so slicing is exact
            SPANS[(basis, r)] = {l: big[l][:, :r].contiguous() for l in EDIT_LAYERS}
        prog(f"  diffpca: one rank-{rmax} PCA per layer, sliced to {ranks} [{time.time()-t0:.0f}s]")
    else:
        for r in ranks:                  # alg1's two-step SVD refines a RANK-DEPENDENT anchor
            SPANS[(basis, r)] = build_layer_subspaces(acts, EDIT_LAYERS, name=basis, rank=r)
        prog(f"  {basis}: rebuilt per rank {ranks} (does not nest) [{time.time()-t0:.0f}s]")

# --- pass 2: layer-major. One whitener per layer, then every cell's edit + per-language e_l. ------
EDITS: dict[tuple, dict[int, object]] = {c: {} for c in CELLS}
ENERGY: dict[tuple, dict[tuple[int, str], float]] = {c: {} for c in CELLS}
_need = ("leace", "wiener", "wiener_marginal")
t0 = time.time()
for i, layer in enumerate(EDIT_LAYERS, 1):
    tl = time.time()
    sl = layer_slice(acts, layer)                       # {lang: (n, d)} views, no copy
    W = layer_whitener(sl, shrink=SHRINK, need=_need)   # the expensive part, once per layer
    for basis, operator, rank in CELLS:
        cell = (basis, operator, rank)
        e = build_family_edits(acts, [layer], basis=basis, operator=operator, rank=rank,
                               shrink=SHRINK, whitener_by_layer={layer: W},
                               span_by_layer=SPANS[(basis, rank)],
                               diffpca_angular=DIFFPCA_ANGULAR)[layer]
        EDITS[cell][layer] = e
        ENERGY[cell].update(edit_energy_by_lang({layer: e}, acts, LANGS,
                                                n_rows=DOSE_ROWS, seed=0))
    del W
    prog(f"  layer {layer:2d} [{i}/{len(EDIT_LAYERS)}]: {len(CELLS)} cells x {len(LANGS)} langs "
         f"[{time.time()-tl:.0f}s; {time.time()-t0:.0f}s elapsed]")

# --- the solve ------------------------------------------------------------------------------------
prog("solving the per-language lambda grid ...")
rows = []
# anchor per family, measured at the CENTRE rank: the lambda axis lives there
ANCHOR: dict[tuple[str, str], float] = {}
CENTRE_DOSE: dict[tuple, dict[str, float]] = {}
for basis, operator, rank in CELLS:
    cell = (basis, operator, rank)
    e_by_lang = {lg: {l: ENERGY[cell][(l, lg)] for l in EDIT_LAYERS} for lg in LANGS}
    CENTRE_DOSE[cell] = {lg: schedule_dose(LAM_ZHAO, e_by_lang[lg]) for lg in LANGS}
    if rank == R0:
        ANCHOR[(basis, operator)] = pool_dose(CENTRE_DOSE[cell].values(), POOL)
        prog(f"  anchor {basis}x{operator}: {POOL} dose at rank {R0} = "
             f"{ANCHOR[(basis, operator)]:.6g}  (per-lang "
             f"{min(CENTRE_DOSE[cell].values()):.4g}..{max(CENTRE_DOSE[cell].values()):.4g})")

n_clamped = 0
for arm in ARMS:
    cell = arm.cell
    e_by_lang = {lg: {l: ENERGY[cell][(l, lg)] for l in EDIT_LAYERS} for lg in LANGS}
    for lang in LANGS:
        d_centre = CENTRE_DOSE[cell][lang]
        if arm.rung == CENTRE_RUNG:
            lam, alpha, clamped = LAM_ZHAO, 1.0, False
            target, source = d_centre, "zhao_global_fixed"
        else:
            target = arm.rung * ANCHOR[(arm.basis, arm.operator)]
            alpha, clamped = solve_alpha(target, d_centre)
            lam = scale_schedule(LAM_ZHAO, alpha)
            source = ("dose_calibrated_from_zhao_global_clamped" if clamped
                      else "dose_calibrated_from_zhao_global")
        n_clamped += int(clamped)
        measured = schedule_dose(lam, e_by_lang[lang])   # RECOMPUTED, not alpha^2 * d_centre
        rows.append({
            "model": TAG, "arm": arm.name, "family": arm.family, "stage": arm.stage,
            "basis": arm.basis, "operator": arm.operator, "rank": arm.rank, "rung": arm.rung,
            "lang": lang,
            "lambda_mid": lam[min(_abl)], "lambda_high": lam[min(_inj)], "alpha": alpha,
            "dose_target": target, "dose_measured": measured, "dose_centre": d_centre,
            "clamped": clamped, "lambda_source": source,
            "dose_anchor": ANCHOR.get((arm.basis, arm.operator), float("nan")),
            "dose_pool": POOL, "n_samples": cfg.n_samples, "dose_rows": DOSE_ROWS or 0,
            "n_langs": N_LANGS,      # the basis is built from every CACHED language; the grid's
                                     # rank cap and arm set derive from it, so the sweep must not
                                     # have to re-infer it from however many langs it happens to run

            "shrink": SHRINK, "n_layers": N_LAYERS, "mid_start": min(_abl), "high_start": min(_inj),
        })

df = pd.DataFrame(rows)
CALIB_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CALIB_CSV, index=False)
prog(f"wrote {CALIB_CSV}  ({len(df)} rows = {len(ARMS)} arms x {len(LANGS)} langs)")
if n_clamped:
    prog(f"WARNING: {n_clamped} (arm, lang) cells CLAMPED — they sit at a dose they were not asked "
         f"for; they are marked lambda_source=*_clamped and must be flagged in the figure")

# exactness check on the solve
off = df[~df.clamped].assign(err=lambda d: (d.dose_measured - d.dose_target).abs()
                            / d.dose_target.clip(lower=1e-30))
worst = float(off.err.max()) if len(off) else 0.0
prog(f"dose solve exactness: worst |measured-target|/target over {len(off)} unclamped cells "
     f"= {worst:.2e}")
assert worst < 1e-6, f"dose solve did not converge: relative error {worst:.3e}"

# --- freeze the edit-sets -------------------------------------------------------------------------
SPEC_DIR.mkdir(parents=True, exist_ok=True)
total = 0
for basis, operator, rank in CELLS:
    cell = (basis, operator, rank)
    path = SPEC_DIR / f"{cell_tag(basis, operator, rank)}.pt"
    torch.save({
        "basis": basis, "operator": operator, "rank": rank, "scope": "prompt_only",
        "n_layers": N_LAYERS, "edit_layers": EDIT_LAYERS, "lam_zhao": LAM_ZHAO,
        "model": cfg.model_name, "tag": TAG, "n_samples": cfg.n_samples, "shrink": SHRINK,
        "diffpca_angular": DIFFPCA_ANGULAR,
        "edits": {int(l): EDITS[cell][l] for l in EDIT_LAYERS},
    }, path)
    total += path.stat().st_size
prog(f"froze {len(CELLS)} edit-sets -> {SPEC_DIR} ({total / 2**20:.0f} MiB)")

# --- what the grid will run ----------------------------------------------------------------------
prog(f"per-family lambda_mid ranges (centre = Zhao's {LAM_ZHAO[min(_abl)]:+.1f} verbatim):")
for fam, g in df.groupby("family", sort=False):
    c = g[g.rung == CENTRE_RUNG]
    o = g[g.rung != CENTRE_RUNG]
    prog(f"  {fam:16s} centre lam_mid={c.lambda_mid.iloc[0]:+.3f} (constant) | "
         f"off-centre {o.lambda_mid.min():+.3f}..{o.lambda_mid.max():+.3f} "
         f"({o.lang.nunique()} langs x {o.rung.nunique()} rungs)")
prog(f"DONE. Next: scripts/e9_tradeoff_grid.py (reads {CALIB_CSV.name} + specs/)")
