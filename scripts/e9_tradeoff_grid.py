#!/usr/bin/env python
"""E9 trade-off grid: the (response fidelity, accuracy) dot cloud. GPU, generation only.

Spec: docs/workshop/tradeoff_grid.md. Run `scripts/e9_dose_calibrate.py` FIRST — this script never
touches the 33 GB activation cache; it reads that step's frozen per-layer edits and per-language
lambda table off disk. Everything geometric is already decided by the time we get here.

WHAT IT PRODUCES. One resumable CSV shard per (language, arm) under
`results/E9/<tag>/e9_grid/<lang>__<cond>.csv`, with the same 15 columns as
`results/E8/items/<model>__<shard>.csv` plus `engine`, so `scripts/e8_joint_metric.py` and
`paper/workshop/figures/make_figs_e8.py` read them unchanged. The arm lives in the **`cond`**
column, not the filename (E8's `__<shard>` suffix is a *language shard*; every consumer globs
`<model>__*.csv` and concatenates).

RESUME. A shard is written last, so a shard on disk is complete; existing shards are skipped. A
walltime kill is resumed by resubmitting the same phase. `E9_REDUCE_ONLY=1` re-reduces whatever is
on disk without loading a model.

STAGE ORDER (tradeoff_grid.md:71). `clean` -> the 6 centre points -> the lambda axis -> the rank
axis, every stage across ALL languages, so each stage is a complete figure on its own and a kill
between stages still leaves something publishable.

ENGINE. `E5_ENGINE=vllm` routes through the production seam in `evaluate.run_eval`; no HF model is
loaded on that path (its VRAM goes to the engine), matching `scripts/validate_engine_parity.py`.
Interventions travel as `intervention_spec=` there, not as forward `hooks=` — we pass BOTH, built
from one spec, so the two engines stay in lock-step and the fail-loud guard is satisfied. The KV
window is sized from the ACTUAL longest prompt and floored at 8192: at 4096 vLLM admits over-long
te prompts and returns zero tokens with finish_reason="length" — silent empties, not an error.

KNOBS
  E5_MODEL=Qwen/Qwen2.5-7B-Instruct   model
  E5_N_SAMPLES=7500                   which calibration to read (stamped in the calib CSV)
  E5_ENGINE=hf|vllm                   decode engine (default hf)
  E9_LANGS=                            comma list; default = every language in the calib table
  E9_ARMS=                             comma list of arm names (or `clean`); default = all
  E9_STAGES=                           comma list of clean,centre,lambda,rank; default = all
  E9_N=                                items per language; default = all 250 (never cut items)
  E9_NSHOT=8                           few-shot exemplars (8 matches the A1 arms)
  E9_DRYRUN=1                          print the job plan and exit
  E9_REDUCE_ONLY=1                     skip generation, reduce existing shards
  E9_LANE=                             lane id, set by the sbatch GPU fan-out; suffixes the
                                       progress log/bar only. Shards stay shared -- lanes
                                       cooperate by skipping each other's finished shards.
  E9_SHARD_DIR=e9_grid                 shard subdirectory under results/E9/<tag>/. The smoke gate
                                       uses its own (e9_grid_smoke_hf / _vllm) so it can never
                                       collide with, or be mistaken for, the real grid's shards.
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

from benchmarks import load_benchmark
from config import Config, get_device, results_path, set_seeds
from dose import scale_schedule
from e9_grid import CLEAN, STAGES, grid_arms, parse_arm
from evaluate import aggregate, run_eval
from intervene import AffineSpec, MaskHolder, build_affine_hooks

set_seeds(0)

cfg = Config(model_name=os.environ.get("E5_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
             n_samples=int(os.environ.get("E5_N_SAMPLES", "7500")))
TAG = cfg.model_name.split("/")[-1]
ENGINE = os.environ.get("E5_ENGINE", "hf").lower()
N_ITEMS = int(os.environ["E9_N"]) if os.environ.get("E9_N") else None      # None -> all 250
N_SHOT = int(os.environ.get("E9_NSHOT", "8"))
DRYRUN = os.environ.get("E9_DRYRUN", "") not in ("", "0")
REDUCE_ONLY = os.environ.get("E9_REDUCE_ONLY", "") not in ("", "0")

CALIB_CSV = results_path(f"E9_calib_{TAG}.csv", TAG)
SPEC_DIR = CALIB_CSV.parent / "e9_specs"
SHARD_NAME = os.environ.get("E9_SHARD_DIR", "e9_grid")
SHARD_DIR = CALIB_CSV.parent / SHARD_NAME
IS_MAIN_GRID = SHARD_NAME == "e9_grid"
_SFX = "" if IS_MAIN_GRID else f"_{SHARD_NAME}"
# One lane per GPU writes into the SAME shard directory -- that is the whole point, since a shard
# on disk is what another lane skips -- but they must not share a progress file, or four writers
# truncate one bar and it reads as a stalled run. E9_LANE is set by the sbatch fan-out; empty for a
# single-process run, which keeps the historical filenames.
_LANE = os.environ.get("E9_LANE", "").strip()
_LN = f"_lane{_LANE}" if _LANE else ""
PROG = results_path(f"E9_grid_progress_{TAG}{_LN}.log", TAG)
PROGBAR = results_path(f"E9_grid_bar_{TAG}{_LN}.txt", TAG)


def prog(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    PROG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def shard_path(lang: str, cond: str) -> pathlib.Path:
    return SHARD_DIR / f"{lang}__{cond}.csv"


# --- the calibration ------------------------------------------------------------------------------
if not CALIB_CSV.exists():
    sys.exit(f"\nMISSING calibration: {CALIB_CSV}\nRun the CPU pre-pass first:\n"
             f"  E5_MODEL={cfg.model_name} E5_N_SAMPLES={cfg.n_samples} "
             f"python scripts/e9_dose_calibrate.py\n")
calib = pd.read_csv(CALIB_CSV)
prog(f"=== E9 trade-off grid | {TAG} | engine={ENGINE} | n/lang={N_ITEMS or 'all(250)'} "
     f"| {N_SHOT}-shot ===")
prog(f"calibration {CALIB_CSV.name}: {len(calib)} rows, {calib.arm.nunique()} arms, "
     f"{calib.lang.nunique()} langs, n_samples={sorted(calib.n_samples.unique())}")
if (calib.n_samples != cfg.n_samples).any():
    sys.exit(f"calibration was built at n_samples={sorted(calib.n_samples.unique())} but this run "
             f"asks for {cfg.n_samples}; re-run the calibration or set E5_N_SAMPLES to match")
n_clamped = int(calib.clamped.astype(str).str.lower().isin(["true", "1"]).sum())
if n_clamped:
    prog(f"NOTE {n_clamped} calibrated (arm, lang) cells are CLAMPED — they sit at a dose they were "
         f"not asked for and carry lambda_source=*_clamped; flag them in the figure")

if "n_langs" not in calib:
    sys.exit(f"{CALIB_CSV.name} has no n_langs column — re-run scripts/e9_dose_calibrate.py "
             f"(the grid's rank cap and arm set derive from it)")
N_LANGS_CACHE = int(calib.n_langs.iloc[0])
ALL_ARMS = {a.name: a for a in grid_arms(N_LANGS_CACHE)}
LANGS = (os.environ["E9_LANGS"].split(",") if os.environ.get("E9_LANGS")
         else sorted(calib.lang.unique()))
WANT_STAGES = (os.environ["E9_STAGES"].split(",") if os.environ.get("E9_STAGES") else list(STAGES))
bad = [s for s in WANT_STAGES if s not in STAGES]
if bad:
    sys.exit(f"unknown stage(s) {bad}; know {list(STAGES)}")

# --- the work plan: stage-major, then arm, then language -----------------------------------------
if os.environ.get("E9_ARMS"):
    want = [a.strip() for a in os.environ["E9_ARMS"].split(",") if a.strip()]
    unknown = [a for a in want if a != CLEAN and a not in ALL_ARMS]
    if unknown:
        sys.exit(f"unknown arm(s) {unknown}\nknown: {CLEAN}, " + ", ".join(sorted(ALL_ARMS)))
    CONDS = want
else:
    CONDS = ([CLEAN] if "clean" in WANT_STAGES else []) + \
            [a.name for a in ALL_ARMS.values() if a.stage in WANT_STAGES]

JOBS = [(lang, cond) for cond in CONDS for lang in LANGS]
TODO = [j for j in JOBS if not shard_path(*j).exists()]
prog(f"plan: {len(CONDS)} conds x {len(LANGS)} langs = {len(JOBS)} shards "
     f"({len(JOBS) - len(TODO)} already on disk, {len(TODO)} to run)")
prog(f"stages {WANT_STAGES} | langs {LANGS}")
prog(f"shards -> {SHARD_DIR}")
if DRYRUN:
    for j in TODO[:16]:
        print("   todo:", j)
    if len(TODO) > 16:
        print(f"   ... +{len(TODO) - 16} more")
    sys.exit(0)

# --- items ----------------------------------------------------------------------------------------
device = get_device()
model = tok = None
items_by_lang: dict[str, list] = {}
# Nothing to generate (a fully resumed job, or REDUCE_ONLY) => never load a model or an engine.
# On a resumed 7B grid job that is minutes of pointless weight loading and a whole card of VRAM.
if REDUCE_ONLY or not TODO:
    prog(f"no generation needed ({'E9_REDUCE_ONLY' if REDUCE_ONLY else 'every shard already on disk'})"
         f" — skipping model/engine load")
else:
    prog("pre-loading MGSM items (warms the dataset cache, fixes the progress total) ...")
    items_by_lang = {l: load_benchmark("mgsm", l, "test", cfg=cfg, n=N_ITEMS, seed=0, n_shot=N_SHOT)
                     for l in LANGS}
    prog("items: " + ", ".join(f"{l}={len(v)}" for l, v in items_by_lang.items()))

    if ENGINE == "vllm":
        # No HF model: run_eval ignores it on this path, so that VRAM goes to the engine. Size the
        # KV window from the ACTUAL longest prompt (te's 8-shot prompt is the long pole on
        # byte-fallback BPEs) and never below 8192 — see the module docstring.
        from transformers import AutoTokenizer

        from corpus import build_chat_inputs
        from models import model_spec
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        longest = 0
        for lang, items in items_by_lang.items():
            enc = tok(build_chat_inputs(tok, [it.prompt for it in items]))["input_ids"]
            longest = max(longest, max(len(x) for x in enc))
        cap = model_spec(TAG).max_new_tokens or 512
        want_len = max(8192, longest + cap + 128)
        os.environ.setdefault("E5_VLLM_MAX_MODEL_LEN", str(want_len))
        prog(f"vLLM: longest prompt {longest} tok + {cap} new -> "
             f"E5_VLLM_MAX_MODEL_LEN={os.environ['E5_VLLM_MAX_MODEL_LEN']} (floor 8192)")
    else:
        from extract import load_model_and_tokenizer
        prog(f"loading HF model on {device} ...")
        model, tok = load_model_and_tokenizer(cfg, device)

# --- specs ----------------------------------------------------------------------------------------
_SPECS: dict[tuple, dict] = {}


def spec_blob(basis: str, operator: str, rank: int) -> dict:
    """The frozen per-layer edits for one (basis, operator, rank) cell, loaded once per process."""
    key = (basis, operator, int(rank))
    if key not in _SPECS:
        path = SPEC_DIR / f"{basis}_{operator}_r{int(rank)}.pt"
        if not path.exists():
            sys.exit(f"\nMISSING frozen edit-set {path}\nRe-run scripts/e9_dose_calibrate.py — it "
                     f"writes one blob per (basis, operator, rank) cell.\n")
        _SPECS[key] = torch.load(path, weights_only=False)
    return _SPECS[key]


def build_spec(cond: str, lang: str) -> AffineSpec:
    """The arm's frozen edits + this language's calibrated lambda, as one AffineSpec."""
    arm = ALL_ARMS[cond]
    blob = spec_blob(arm.basis, arm.operator, arm.rank)
    row = calib[(calib.arm == cond) & (calib.lang == lang)]
    if len(row) != 1:
        sys.exit(f"calibration has {len(row)} rows for (arm={cond}, lang={lang}); expected exactly 1")
    row = row.iloc[0]
    # Rebuild lambda exactly as the calibrator did — rescale Zhao's schedule by the stored alpha —
    # rather than re-deriving band boundaries here, then cross-check against the logged value.
    lam_zhao = {int(k): float(v) for k, v in blob["lam_zhao"].items()}
    lam = scale_schedule(lam_zhao, float(row.alpha))
    mid = min(l for l, v in lam_zhao.items() if v > 0)
    if abs(lam[mid] - float(row.lambda_mid)) > 1e-9:
        sys.exit(f"spec/calib disagree for {cond}/{lang}: rebuilt lambda_mid={lam[mid]:.9f} vs "
                 f"logged {float(row.lambda_mid):.9f} — the spec blob and calib CSV are out of sync")
    return AffineSpec(edits={int(k): v for k, v in blob["edits"].items()}, lam=lam,
                      scope="prompt_only")


# --- generation ----------------------------------------------------------------------------------
if TODO and not REDUCE_ONLY:
    N_TOTAL = sum(len(items_by_lang[l]) for l, _ in TODO)
    prog(f"{N_TOTAL} generations remaining across {len(TODO)} shards")
    _done, _t0 = 0, time.time()

    def _hms(s: float) -> str:
        if s != s:
            return "--"
        s = int(s)
        return f"{s//3600}h{(s % 3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s % 60:02d}s"

    def _bar(cur: str) -> None:
        frac = _done / N_TOTAL if N_TOTAL else 0.0
        el = time.time() - _t0
        eta = (el / frac - el) if frac > 0.02 else float("nan")
        filled = int(round(28 * frac))
        with open(PROGBAR, "w") as f:
            f.write(f"[{'█' * filled}{'░' * (28 - filled)}] {100*frac:5.1f}%  {_done}/{N_TOTAL} gens"
                    f"  @{cur}  elapsed {_hms(el)}  ETA {_hms(eta)}\n")

    _bar("-")
    for lang, cond in TODO:
        out = shard_path(lang, cond)
        if out.exists():                       # a sibling job may have filled it
            continue
        items = items_by_lang[lang]
        t0 = time.time()
        _bar(f"{lang}/{cond}")
        if cond == CLEAN:
            hooks, mh, spec = None, None, None
        else:
            spec = build_spec(cond, lang)
            mh = MaskHolder(scope="prompt_only")
            hooks = build_affine_hooks(spec, mask_holder=mh)
        # Pass hooks AND intervention_spec: the HF path uses the hooks, vLLM uses the spec, and the
        # fail-loud guard in run_eval rejects hooks-without-spec (silently-clean "intervened" rows).
        df = run_eval(model, tok, items, hooks=hooks, mask_holder=mh, intervention_spec=spec,
                      device=device, model_name=cfg.model_name, cond=cond)
        parsed = parse_arm(cond) or {}
        for k, v in parsed.items():
            df[k] = v
        row = calib[(calib.arm == cond) & (calib.lang == lang)]
        for col in ("lambda_mid", "lambda_high", "alpha", "dose_target", "dose_measured",
                    "dose_centre", "clamped", "lambda_source"):
            df[col] = (row.iloc[0][col] if len(row) == 1 else
                       ("clean" if col == "lambda_source" else 0.0 if col != "clamped" else False))
        df["n_shot"] = N_SHOT
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)             # written last => a shard on disk is complete
        _done += len(items)
        det = df[~df["output_lang"].astype(str).isin(["", "?"])]
        fid = float((det["output_lang"] == lang).mean()) if len(det) else float("nan")
        prog(f"{lang:3s} {cond:28s} acc={df['correct'].mean():.3f} fid={fid:.3f} "
             f"trunc={df['truncated'].mean():.3f} [{time.time()-t0:.0f}s]")
        _bar(f"{lang}/{cond}")

# --- reduce ---------------------------------------------------------------------------------------
shards = sorted(SHARD_DIR.glob("*.csv")) if SHARD_DIR.exists() else []
if not shards:
    sys.exit("no shards on disk — nothing to reduce")
prog(f"reducing {len(shards)} shards ...")
items_df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
agg_df = pd.concat([aggregate(items_df[items_df["cond"] == c]) for c in items_df["cond"].unique()],
                   ignore_index=True)
# Written directly rather than through evaluate.write_results, which hardcodes an `E5_` filename
# prefix (that would name these E5_E9_grid_*.csv).
item_path = results_path(f"E9_grid_items{_SFX}_{TAG}.csv", TAG)
agg_path = results_path(f"E9_grid_agg{_SFX}_{TAG}.csv", TAG)
items_df.to_csv(item_path, index=False)
agg_df.to_csv(agg_path, index=False)

# The dot coordinates, one row per (arm, language) — this IS the figure.
items_df["fid"] = (items_df.output_lang.astype(str) == items_df.lang).astype(float)
dots = (items_df.groupby(["cond", "lang"], sort=False)
        .agg(acc=("correct", "mean"), fid=("fid", "mean"),
             trunc=("truncated", "mean"), n=("correct", "size")).reset_index())
meta = items_df.drop_duplicates(["cond", "lang"])[
    ["cond", "lang"] + [c for c in ("family", "basis", "operator", "rank", "rung", "lambda_mid",
                                    "dose_target", "dose_measured", "clamped", "lambda_source",
                                    "engine") if c in items_df]]
dots = dots.merge(meta, on=["cond", "lang"], how="left")
dots_path = results_path(f"E9_grid_dots{_SFX}_{TAG}.csv", TAG)
dots.to_csv(dots_path, index=False)

# The figure-facing items dir, so E9_ITEMS_DIR=results/E9/items works with make_figs_e8.py
# unchanged. Only the real grid publishes here — a smoke run's 1-language shards must not become
# the figure, and must not overwrite a completed grid's copy.
wrote_fig = ""
if IS_MAIN_GRID:
    fig_dir = CALIB_CSV.parent.parent / "items"
    fig_dir.mkdir(parents=True, exist_ok=True)
    items_df.to_csv(fig_dir / f"{TAG}__all.csv", index=False)
    wrote_fig = f", items/{TAG}__all.csv"

prog(f"wrote {item_path.name}, {agg_path.name}, {dots_path.name}{wrote_fig}")
prog(f"engines present: {sorted(items_df.engine.unique()) if 'engine' in items_df else ['<none>']}")
for cond, g in dots.groupby("cond", sort=False):
    prog(f"  {cond:28s} acc={g.acc.mean():.3f} fid={g.fid.mean():.3f} "
         f"({g.lang.nunique()} langs, n={int(g.n.sum())})")
