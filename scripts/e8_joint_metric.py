#!/usr/bin/env python
"""A1 -- the fidelity-locked results table (CPU only, no model, no GPU).

Task A1 of docs/workshop/proposal_and_plan.md. Reads the E6 phase-2 per-item CSVs and produces
the paper's results section: accuracy, response fidelity, and the JOINT metric
(correct AND answered-in-the-target-language) per model x arm x language, with

  * problem-cluster bootstrap CIs -- MGSM is parallel, so one problem index is ONE cluster
    across all languages in a pool (verified: the 125 problem indices are identical across all
    11 languages). Resampling rows instead of problems understates the interval.
  * paired deltas vs `clean` and vs `alg1_tuned`, bootstrapped on the same resampled problems
    so the pairing is preserved.
  * Holm correction across the arm family, within each (model, pool, contrast).
  * both undetectable-output conventions: fid_strict counts '?'/NaN as a miss, fid_repo excludes
    it from the denominator (`scoring.detlang` / `evaluate.aggregate` behaviour).
  * a truncation gate -- a delta between arms whose truncation rates differ by more than
    E8_TRUNC_EPS is flagged confounded rather than silently reported.

Pools are macro-means over languages (the unweighted per-language mean, matching Zhao Table 1's
AVG), not item-weighted means.

Input:  a directory of per-item CSVs named <model>__<shard>.csv (see e8_joint_metric_smoke.py
        for the git-show recipe that populates it).
Output: results/E8/joint_perlang_<model>.csv   -- arm x lang rows, no CIs (per-cell n=125)
        results/E8/joint_pooled_<model>.csv    -- arm x pool rows, CIs + Holm + gate
        results/E8/joint_summary.csv           -- one row per model x arm x pool, all models
        stdout: the tables as read in the paper

Usage:
    python scripts/e8_joint_metric.py
    E8_BOOT=4000 E8_TRUNC_EPS=0.05 python scripts/e8_joint_metric.py
    # E9: its own items dir, its own centre arm as a second reference, and te admitted
    E8_ITEMS_DIR=results/E9/items E8_LOW=bn,sw,te,th \
      E8_REFS=clean,e9_alg1_proj_r10_d100 python scripts/e8_joint_metric.py
"""
from __future__ import annotations

import glob
import os
import pathlib

import numpy as np
import pandas as pd

ITEMS_DIR = os.environ.get("E8_ITEMS_DIR", "results/E8/items")
OUT_DIR = pathlib.Path(os.environ.get("E8_OUT_DIR", "results/E8"))
N_BOOT = int(os.environ.get("E8_BOOT", "4000"))
TRUNC_EPS = float(os.environ.get("E8_TRUNC_EPS", "0.05"))
SEED = int(os.environ.get("E8_SEED", "0"))

# Language pools, overridable because which languages are *usable* depends on the campaign.
# `te` is excluded from the LOW default, but the reason is historical and narrower than it looks:
# on the pre-fix vLLM path `max_model_len=4096` admitted the over-long Telugu prompt and returned
# zero tokens, so the A1 per-item CSVs carry te at acc 0.000 / fid 0.000 / trunc 1.000 -- dead data
# that would drag any pool containing it. That bug is fixed (CLAUDE.md Corrections #4): E9 measures
# te at acc 0.136 / fid 1.000, real but still truncation-pressed at 0.788 against a 512-token cap.
# So: keep the default for the A1 tables, and pass `E8_LOW=bn,sw,te,th` for E9 or any post-fix run.
HIGH = [l.strip() for l in os.environ.get("E8_HIGH", "de,en,es,fr,ja,ru,zh").split(",") if l.strip()]
LOW = [l.strip() for l in os.environ.get("E8_LOW", "bn,sw,th").split(",") if l.strip()]
# The combined pool is named for how many languages actually survive the filter below, so the label
# can never claim a breadth the rows do not have (`all10` on A1, `all11` once te is admitted).
POOLS = {"high": HIGH, "low": LOW, "all": HIGH + LOW}
# Contrasts every arm is scored against. Overridable because the reference depends on the campaign:
# A1 compares each operator to `clean` and to the projection baseline `alg1_tuned`, while E9's
# baseline is `clean` plus its own centre arm (E8_REFS=clean,e9_alg1_proj_r10_d100). A ref that is
# absent from the data is skipped, so a wrong entry is inert rather than fatal.
REFS = [r.strip() for r in os.environ.get("E8_REFS", "clean,alg1_tuned").split(",") if r.strip()]
UNDET = {"?", "nan", "none", ""}

ARM_ORDER = [
    "clean", "alg1_tuned", "diffpca_tuned", "diffpca_raw_tuned",
    "leace_tuned", "wiener_tuned", "shrink_tuned",
    "diffpca_wiener_tuned", "diffpca_oblique_tuned",
    "diffpca_raw_wiener_tuned", "diffpca_raw_oblique_tuned",
]


# ---------------------------------------------------------------- load / derive

def load_model(model: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{ITEMS_DIR}/{model}__*.csv"))
    if not files:
        raise FileNotFoundError(f"no item CSVs for {model} under {ITEMS_DIR}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    d["prob"] = d["id"].str.extract(r"-(\d+)$").astype(int)
    ol = d["output_lang"].astype(str).str.strip().str.lower()
    d["undet"] = d["output_lang"].isna() | ol.isin(UNDET)
    d["correct"] = d["correct"].astype(float)
    d["in_lang"] = (d["output_lang"] == d["lang"]).astype(float)
    # strict: undetectable is a miss. repo: undetectable drops out of the denominator.
    d["fid_strict"] = d["in_lang"]
    d["fid_repo"] = np.where(d["undet"], np.nan, d["in_lang"])
    d["joint"] = (d["correct"].astype(bool) & d["in_lang"].astype(bool)).astype(float)
    d["trunc"] = d["truncated"].astype(float)

    dup = d.duplicated(["cond", "lang", "prob"]).sum()
    if dup:
        raise ValueError(f"{model}: {dup} duplicate (cond,lang,prob) rows -- shards overlap?")
    return d


def arms_of(d: pd.DataFrame) -> list[str]:
    """Arms present, in `ARM_ORDER` first, then anything unlisted, sorted.

    The tail matters: E9's 37 parameterized arms are not in `ARM_ORDER` and must still be scored.
    Sorting `e9_<basis>_<operator>_r<rank>_d<rung>` groups them by family then rank then dose, which
    is the order the trade-off grid reads in.
    """
    present = set(d["cond"].unique())
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


# ---------------------------------------------------------------- bootstrap core

def cube(d: pd.DataFrame, metric: str, arms: list[str], langs: list[str]) -> np.ndarray:
    """(arm, lang, prob) array of the metric; NaN where absent. Problem axis is shared, which
    is what makes the pairing and the cluster resampling valid."""
    probs = np.sort(d["prob"].unique())
    pi = {p: i for i, p in enumerate(probs)}
    out = np.full((len(arms), len(langs), len(probs)), np.nan)
    sub = d[d["cond"].isin(arms) & d["lang"].isin(langs)]
    ai = {a: i for i, a in enumerate(arms)}
    li = {l: i for i, l in enumerate(langs)}
    out[sub["cond"].map(ai).to_numpy(),
        sub["lang"].map(li).to_numpy(),
        sub["prob"].map(pi).to_numpy()] = sub[metric].to_numpy()
    return out


def macro(cb: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Pooled macro-mean per arm: mean over problems (resampled), then mean over languages."""
    with np.errstate(invalid="ignore"):
        per_lang = np.nanmean(cb[:, :, idx], axis=2)   # (arm, lang)
        return np.nanmean(per_lang, axis=1)            # (arm,)


def boot_pool(cb: np.ndarray, n_probs: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, arm) macro-means under problem-cluster resampling."""
    draws = rng.integers(0, n_probs, size=(N_BOOT, n_probs))
    return np.stack([macro(cb, draws[b]) for b in range(N_BOOT)])


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def two_sided_p(delta_boot: np.ndarray) -> float:
    """Bootstrap two-sided p for H0: delta == 0."""
    n = np.count_nonzero(~np.isnan(delta_boot))
    if not n:
        return np.nan
    le = np.nanmean(delta_boot <= 0)
    ge = np.nanmean(delta_boot >= 0)
    return float(min(1.0, 2 * min(le, ge)))


# ---------------------------------------------------------------- per-model pass

def run_model(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = load_model(model)
    arms = arms_of(d)
    rng = np.random.default_rng(SEED)

    # ---- per-language table (descriptive; n=125 per cell, no CI)
    g = d.groupby(["cond", "lang"])
    per_lang = g.agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        fid_strict=("fid_strict", "mean"),
        fid_repo=("fid_repo", "mean"),
        joint=("joint", "mean"),
        undet_rate=("undet", "mean"),
        trunc_rate=("trunc", "mean"),
    ).reset_index()
    per_lang.insert(0, "model", model)
    per_lang["cond"] = pd.Categorical(per_lang["cond"], arms, ordered=True)
    per_lang = per_lang.sort_values(["cond", "lang"])

    # ---- pooled table with cluster-bootstrap CIs
    n_probs = d["prob"].nunique()
    rows = []
    for pool, langs in POOLS.items():
        langs = [l for l in langs if l in set(d["lang"])]
        if pool == "all":
            pool = f"all{len(langs)}"
        cubes = {m: cube(d, m, arms, langs) for m in ("correct", "fid_strict", "joint", "trunc")}
        boots = {m: boot_pool(cubes[m], n_probs, np.random.default_rng(SEED))
                 for m in ("correct", "fid_strict", "joint")}
        point = {m: macro(cubes[m], np.arange(n_probs)) for m in cubes}

        pending: dict[str, dict[str, float]] = {}   # (metric, ref) -> {arm: p}
        staged: dict[tuple[str, str, str], dict] = {}

        for ai, arm in enumerate(arms):
            row = {"model": model, "pool": pool, "cond": arm, "n_langs": len(langs),
                   "n_probs": n_probs}
            for m in ("correct", "fid_strict", "joint"):
                lo, hi = np.nanpercentile(boots[m][:, ai], [2.5, 97.5])
                row[m] = point[m][ai]
                row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
            row["trunc_rate"] = point["trunc"][ai]
            rows.append(row)

            for ref in REFS:
                if ref not in arms or arm == ref:
                    continue
                ri = arms.index(ref)
                gate = abs(point["trunc"][ai] - point["trunc"][ri])
                for m in ("correct", "fid_strict", "joint"):
                    db = boots[m][:, ai] - boots[m][:, ri]
                    p = two_sided_p(db)
                    lo, hi = np.nanpercentile(db, [2.5, 97.5])
                    staged[(arm, ref, m)] = {
                        "delta": point[m][ai] - point[m][ri],
                        "lo": lo, "hi": hi, "p": p,
                        "trunc_gap": gate,
                        "confounded": bool(gate > TRUNC_EPS),
                    }
                    pending.setdefault(f"{m}|{ref}", {})[arm] = p

        adj = {fam: holm(ps) for fam, ps in pending.items()}
        for (arm, ref, m), rec in staged.items():
            r = next(x for x in rows if x["cond"] == arm and x["pool"] == pool)
            r[f"d_{m}_vs_{ref}"] = rec["delta"]
            r[f"d_{m}_vs_{ref}_lo"] = rec["lo"]
            r[f"d_{m}_vs_{ref}_hi"] = rec["hi"]
            r[f"d_{m}_vs_{ref}_p"] = rec["p"]
            r[f"d_{m}_vs_{ref}_p_holm"] = adj[f"{m}|{ref}"][arm]
            r[f"d_{m}_vs_{ref}_confounded"] = rec["confounded"]
            r[f"trunc_gap_vs_{ref}"] = rec["trunc_gap"]

    pooled = pd.DataFrame(rows)
    pooled["cond"] = pd.Categorical(pooled["cond"], arms, ordered=True)
    pooled = pooled.sort_values(["pool", "cond"])
    return per_lang, pooled


# ---------------------------------------------------------------- reporting

def star(p: float) -> str:
    if not np.isfinite(p):
        return "  "
    return "**" if p < 0.01 else ("* " if p < 0.05 else "  ")


def report(model: str, pooled: pd.DataFrame) -> None:
    print("=" * 118)
    print(f"{model}    (problem-cluster bootstrap, B={N_BOOT}, Holm within arm family)")
    print("=" * 118)
    for pool in ("high", "low"):
        p = pooled[pooled["pool"] == pool]
        if p.empty:
            continue
        print(f"\n  pool={pool}  ({int(p.n_langs.iloc[0])} langs x {int(p.n_probs.iloc[0])} problems)")
        print(f"  {'arm':<27}{'acc':>7}{'fid':>7}{'joint':>7}   "
              f"{'Δjoint vs clean [95% CI]':>30}{'holm':>8}  {'gate':>5}")
        print("  " + "-" * 114)
        for _, r in p.iterrows():
            if r["cond"] == "clean":
                print(f"  {r['cond']:<27}{r['correct']:>7.3f}{r['fid_strict']:>7.3f}"
                      f"{r['joint']:>7.3f}   {'(reference)':>30}{'':>8}")
                continue
            d, lo, hi = r.get("d_joint_vs_clean"), r.get("d_joint_vs_clean_lo"), r.get("d_joint_vs_clean_hi")
            ph = r.get("d_joint_vs_clean_p_holm", np.nan)
            ci = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            flag = "TRUNC" if r.get("d_joint_vs_clean_confounded") else ""
            print(f"  {r['cond']:<27}{r['correct']:>7.3f}{r['fid_strict']:>7.3f}"
                  f"{r['joint']:>7.3f}   {ci:>30}{ph:>8.3f}{star(ph)}{flag:>5}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{ITEMS_DIR}/*.csv")})
    if not models:
        raise SystemExit(f"no item CSVs under {ITEMS_DIR}/ -- see the docstring for the recipe")

    all_pooled = []
    for m in models:
        per_lang, pooled = run_model(m)
        per_lang.to_csv(OUT_DIR / f"joint_perlang_{m}.csv", index=False)
        pooled.to_csv(OUT_DIR / f"joint_pooled_{m}.csv", index=False)
        all_pooled.append(pooled)
        report(m, pooled)

    summary = pd.concat(all_pooled, ignore_index=True)
    summary.to_csv(OUT_DIR / "joint_summary.csv", index=False)
    print(f"\nwrote {OUT_DIR}/joint_{{perlang,pooled}}_<model>.csv and joint_summary.csv "
          f"({len(summary)} pooled rows, {len(models)} models)")


if __name__ == "__main__":
    main()
