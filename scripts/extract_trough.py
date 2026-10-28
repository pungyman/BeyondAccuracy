"""Extract the E3 ρ trough into a machine-readable JSON hand-off.

Takes the argmin of the per-prompt captured fraction over the `alg1` subspace — the exact rule
that seeded `tuning.TROUGH_LAYER` from the committed n=512 runs. Writes
`results/shared/trough_<tag>.json`, the file `tuning.trough_layer` consumes via the
`E5_TROUGH_JSON` env var (set per model by e5_scaleout.py) and that `scripts/e10_probe_erasure.py`
reads directly — E10 exits 3 without it.

## Two accepted sources, tried in order

1. `E3_headline_by_layer_<tag>[_n<n>].csv` — notebook 04's output, the original source.
2. `E3_rank_sweep_math_<tag>_n<n>.csv` — `scripts/e3_rank_sweep.py` mode=math (Version A).

The fallback exists because a new model added after 2026-08-31 gets the rank sweep and **not**
the notebook, and E10 was therefore unreachable for it: `E10_TROUGH` is read from the environment,
but every Prajna job runs `--export=NONE` and `e10_prajna.sbatch` takes no trough argument, so
there was no way to hand E10 a layer at all. Version A is a strict extension of the committed
protocol (same 50/50 split, same seed), so its rank-10 slice is the same quantity the headline CSV
carries — verified on Qwen2.5-7B, where the two agree on all 28 layers to four decimals.

**The rank filter is not optional.** A rank-sweep CSV holds every rank on the ladder, and ρ falls
with rank, so an unfiltered argmin returns the rank-1 row and can name the wrong layer entirely.
`--rank` (default 10, the protocol rank) is applied whenever the CSV has a `rank` column.

The JSON name is un-n-stamped on purpose: it is "the latest measured trough for this model"; the
`n`, `rank` and `source_csv` fields inside record where it came from.

Usage:
    python scripts/extract_trough.py --model Qwen/Qwen2.5-7B-Instruct --n 7500
    python scripts/extract_trough.py --model Qwen/Qwen3-8B --n 7500        # rank-sweep fallback
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

import pandas as pd

_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
             if (p / "src" / "config.py").exists())
sys.path.insert(0, str(_root / "src"))
from config import results_path  # noqa: E402
from trough import depth, describe, interior_trough  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="E3 headline CSV -> results/shared/trough_<tag>.json")
    ap.add_argument("--model", required=True, help="HF model id or bare model tag.")
    ap.add_argument("--n", type=int, default=512, help="Corpus size the E3 sweep ran at.")
    ap.add_argument("--csv", type=pathlib.Path, default=None,
                    help="Override the source CSV path (default: headline, then rank sweep).")
    ap.add_argument("--rank", type=int, default=10,
                    help="Rank slice to read when the CSV has a `rank` column (default 10).")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="Override the output JSON path (default: results/shared/trough_<tag>.json).")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    run_tag = tag if args.n == 512 else f"{tag}_n{args.n}"      # nb04 cell-1 RUN_TAG rule
    if args.csv:
        csv = args.csv
    else:
        candidates = [results_path(f"E3_headline_by_layer_{run_tag}.csv", tag, mkdir=False),
                      results_path(f"E3_rank_sweep_math_{tag}_n{args.n}.csv", tag, mkdir=False)]
        csv = next((c for c in candidates if c.exists()), None)
        if csv is None:
            print("[extract_trough] ERROR: no source CSV. Tried:", file=sys.stderr)
            for c in candidates:
                print(f"    {c}", file=sys.stderr)
            print(f"  Run notebook 04, or the E3 `sweep` phase, for (model={tag}, n={args.n}).",
                  file=sys.stderr)
            return 1
    if not csv.exists():
        print(f"[extract_trough] ERROR: {csv} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(csv)
    if "rho_perprompt" not in df.columns:
        print(f"[extract_trough] ERROR: {csv} has no rho_perprompt column", file=sys.stderr)
        return 1
    alg1 = df[df["subspace"] == "alg1"]
    # A rank-sweep CSV carries the whole ladder and rho falls with rank, so an unfiltered argmin
    # returns the rank-1 row and can name the wrong layer. Slice to the protocol rank first.
    rank_used = None
    if "rank" in df.columns:
        rank_used = args.rank
        alg1 = alg1[alg1["rank"] == rank_used]
        if alg1.empty:
            have = sorted(df[df["subspace"] == "alg1"]["rank"].unique())
            print(f"[extract_trough] ERROR: {csv} has no alg1 rows at rank {rank_used} "
                  f"(present: {have})", file=sys.stderr)
            return 1
    if alg1.empty:
        print(f"[extract_trough] ERROR: {csv} has no alg1 rows", file=sys.stderr)
        return 1
    # Interior-guarded. A global argmin returns layer 1 on the post-fix Qwen3-8B cache, which
    # would seed E10's intervention band at 3% depth. See src/trough.py and CLAUDE.md #9.
    t_layer, t_value, g_layer, g_value = interior_trough(alg1["layer"].tolist(),
                                                         alg1["rho_perprompt"].tolist())
    row = alg1[alg1["layer"] == t_layer].iloc[0]

    out = args.out or results_path(f"trough_{tag}.json", tag)
    payload = {
        "model": args.model,
        "model_tag": tag,
        "n": args.n,
        "trough_layer": int(t_layer),
        "rho_min": float(t_value),
        "relative_depth": round(depth(t_layer, alg1["layer"].tolist()), 4),
        "global_argmin_layer": int(g_layer),
        "subspace": "alg1",
        "rank": rank_used,
        "column": "rho_perprompt",
        "source_csv": csv.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[extract_trough] {tag}: "
          f"{describe(t_layer, t_value, g_layer, g_value, alg1['layer'].tolist())}"
          f" | n={args.n}{'' if rank_used is None else f', rank {rank_used}'}"
          f" from {csv.name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
