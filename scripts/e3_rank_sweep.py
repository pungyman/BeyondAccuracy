"""E3 rho-vs-layer, with **rank** promoted from a fixed constant to a swept axis.

The committed E3 sweep (notebooks/04_E3_residuals.ipynb) fixes the rank at `cfg.rank` = L-1 = 10
and reports rho(layer) for five subspaces. This script reports rho(layer, rank), in two
evaluation modes that answer different questions:

  math   fit on the TRAIN half of the MATH cache, evaluate held-out translation vectors from the
         TEST half. A strict extension of the committed protocol (same 50/50 problem split, same
         seed), so its rank-10 slice must reproduce the committed numbers -- that reproduction is
         the correctness gate, printed at the end.

  mgsm   fit on ALL of the MATH cache, evaluate translation vectors built from MGSM activations
         (scripts/e3_mgsm_acts.py). Cross-corpus: the subspace never sees the benchmark every
         intervention number is measured on, so a trough that survives here is a property of the
         model rather than of the fitting corpus.

The two modes are NOT on a common scale. MGSM vectors are out of the fitting domain and a
different register, so the absolute rho level will sit lower in `mgsm` and that offset means
nothing on its own. What transfers is the shape over depth and the ordering across ranks.

## Why the rank axis is nearly free

rho(v) = ||M^T v||^2 / ||v||^2 = sum_i (u_i . v)^2 over the orthonormal columns u_i. For a
subspace whose columns are NESTED in rank -- PCA components, and the first r columns of a Haar
frame -- the rank-r rho is a prefix sum of the rank-R coordinates. So `diffpca`, `pooled` and
`random` are fit ONCE at the maximum rank and every smaller rank falls out of a cumulative sum.

`alg1` is not nested: its anchor refinement runs through a rank-r truncation of W, so the rank-r
subspace is not the first r columns of the rank-R one. It is refit per rank, which costs nothing
-- an SVD of a (d, L=11) matrix. `none` is refit alongside it for the same reason.

## The two things that make the plot honest

- **alg1 and `none` stop at L-1 = 10 by construction.** `build_language_subspace` takes the top
  `rank` left singular vectors of a matrix with L columns, so past L-1 there are no further real
  directions. Ranks above the cap are not reported for those subspaces (they would be numerical
  noise, and plotting noise next to a real diffpca curve invites the wrong reading).
- **The random floor rises with rank** (rho ~ r/d: 0.003 at r=10, 0.028 at r=100 for d=3584).
  Raw rho-vs-rank therefore flatters high rank for free. Every row carries `rho_floor` (the
  measured `random` value at that rank) and `rho_excess` (rho - rho_floor); read the excess.

## The per-pair side file

The sweep already computes rho for each of the 55 unordered language pairs and averages them away
before writing. That average hides the thing the 2026-08-31 read of the L18->L19 step needs: at the
trough, the committed E3.2 pair heatmap shows a bright CROSS -- pairs touching `en` or `zh` stay
well captured while pairs between two other languages collapse. Read off a PNG that is a hunch;
this file makes it a number.

So a second CSV, `E3_pairrho_<mode>_<tag>_n<N>.csv`, carries one row per
(layer, subspace, rank, pair) -- same compute, ~55x the rows (~52k, still trivial). The **primary
CSV keeps its exact shape**, because paper/workshop/figures/make_figs_e3rank.py reads it. The name
deliberately does NOT start with `E3_rank_sweep_`, since that script globs on that prefix.

Note this is not an artifact of Algorithm 1's anchor: `build_language_subspace` anchors on
`W @ ones / L`, the UNIFORM mean over all 11 languages, so no language is privileged by
construction. `diffpca` has no anchor at all and is the control.

Environment knobs (set by scripts/prajna/e3_prajna.sbatch; --export=NONE means your shell's
exports never reach the job):

  LRD_MODEL         HF model id                                (default: config default)
  LRD_N_SAMPLES     construction-corpus size                   (default 7500)
  E3R_MODE          math | mgsm                                (default math)
  E3R_RANKS         comma-separated rank ladder                (default 1,2,3,5,10,20,50,100)
  E3R_LAYERS        comma-separated layers, or "all"           (default all)
  E3R_MGSM_ACTS     path to the MGSM activation cache          (default: derived from the tag)
  E3R_SEED          problem-split seed                         (default 0)
  E3R_PAIRS         write the per-pair side file               (default 1; set 0 to skip)
  E3R_PIVOTS        pivot languages for the read-out           (default en,zh)

Usage:
    python scripts/e3_rank_sweep.py
    E3R_MODE=mgsm python scripts/e3_rank_sweep.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
             if (p / "src" / "config.py").exists())
sys.path.insert(0, str(_root / "src"))

from candidates import diffpca_subspace                                  # noqa: E402
from config import ACTS_DIR, Config, LINALG_DEVICE, results_path, set_seeds  # noqa: E402
from controls import pooled_pca_subspace, random_orthonormal_subspace    # noqa: E402
from extract import acts_cache_path                                      # noqa: E402
from geometry import ordered_or_unordered_pairs, split_problems          # noqa: E402
from subspace import build_language_subspace, build_subspace_variant, language_means  # noqa: E402
from trough import describe, interior_trough                            # noqa: E402

# Nested in rank (prefix sums valid) vs refit per rank.
NESTED = ("diffpca", "pooled", "random")
REFIT = ("alg1", "none")
SUBSPACES = REFIT + NESTED


def _env_ranks() -> list[int]:
    raw = os.environ.get("E3R_RANKS", "1,2,3,5,10,20,50,100")
    return sorted({int(r) for r in raw.split(",") if r.strip()})


def _coords(V: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Per-vector squared coordinates ``(v . u_i)^2`` on an orthonormal basis. `(m, r)`."""
    return (V.to(LINALG_DEVICE, torch.float32) @ M.to(LINALG_DEVICE, torch.float32)).pow(2)


def _rho_at_ranks(V: torch.Tensor, M_max: torch.Tensor, ranks: list[int]) -> dict[int, float]:
    """Mean rho over the rows of `V` at each rank, by prefix sum over a NESTED basis."""
    V = V.to(LINALG_DEVICE, torch.float32)
    den = (V * V).sum(dim=1).clamp_min(1e-12)
    cum = _coords(V, M_max).cumsum(dim=1)                   # (m, R)
    return {r: float((cum[:, r - 1] / den).mean()) for r in ranks if r <= M_max.shape[1]}


def _rho_single(V: torch.Tensor, M: torch.Tensor) -> float:
    """Mean rho over the rows of `V` for one (non-nested) basis."""
    V = V.to(LINALG_DEVICE, torch.float32)
    den = (V * V).sum(dim=1).clamp_min(1e-12)
    return float((_coords(V, M).sum(dim=1) / den).mean())


def main() -> int:
    t_start = time.time()
    mode = os.environ.get("E3R_MODE", "math").strip().lower()
    if mode not in ("math", "mgsm"):
        print(f"ERROR: E3R_MODE must be math|mgsm, got {mode!r}", file=sys.stderr)
        return 2

    seed = int(os.environ.get("E3R_SEED", "0"))
    set_seeds(seed)
    cfg = Config(model_name=os.environ.get("LRD_MODEL", Config.model_name),
                 n_samples=int(os.environ.get("LRD_N_SAMPLES", "7500")))
    tag = cfg.model_name.split("/")[-1]
    langs = list(cfg.langs)
    L = len(langs)
    cap = L - 1                                             # alg1 / none structural rank cap

    ranks = _env_ranks()
    fit_rank = max(ranks)

    # ---- fit corpus: the MATH activation cache -------------------------------------------------
    cache = acts_cache_path(cfg)
    if not cache.exists():
        print(f"ERROR: activation cache missing: {cache}", file=sys.stderr)
        return 3
    print(f"[{time.time()-t_start:7.1f}s] loading fit cache {cache}", flush=True)
    acts = torch.load(cache, map_location="cpu", weights_only=True)
    n_samples, n_layers, d = acts[langs[0]].shape
    print(f"           fit acts: {L} langs x ({n_samples} problems, {n_layers} layers, d={d})",
          flush=True)

    if fit_rank > d:
        print(f"ERROR: max rank {fit_rank} exceeds d={d}", file=sys.stderr)
        return 2

    # ---- evaluation source ---------------------------------------------------------------------
    # math: fit on the train half, evaluate held-out vectors from the test half (committed protocol).
    # mgsm: fit on everything, evaluate vectors from a different corpus entirely.
    if mode == "math":
        train_idx, test_idx = split_problems(n_samples, frac=0.5, seed=seed)
        fit_sel = torch.as_tensor(train_idx)
        eval_acts = {l: acts[l][torch.as_tensor(test_idx)] for l in langs}
        eval_note = f"MATH held-out ({len(test_idx)} problems, seed {seed})"
    else:
        mgsm_path = pathlib.Path(os.environ.get(
            "E3R_MGSM_ACTS", ACTS_DIR / f"mgsm_acts_{tag}_nshot0.pt"))
        if not mgsm_path.exists():
            print(f"ERROR: MGSM activation cache missing: {mgsm_path}\n"
                  f"       run the 'mgsmacts' phase for this model first.", file=sys.stderr)
            return 3
        print(f"[{time.time()-t_start:7.1f}s] loading MGSM acts {mgsm_path}", flush=True)
        mg = torch.load(mgsm_path, map_location="cpu", weights_only=True)
        missing = [l for l in langs if l not in mg]
        if missing:
            print(f"ERROR: MGSM acts missing languages {missing}", file=sys.stderr)
            return 3
        if mg[langs[0]].shape[1] != n_layers or mg[langs[0]].shape[2] != d:
            print(f"ERROR: MGSM acts shape {tuple(mg[langs[0]].shape)} does not match the fit "
                  f"cache (layers={n_layers}, d={d}) — different model?", file=sys.stderr)
            return 3
        fit_sel = torch.arange(n_samples)                   # fit on ALL of MATH
        eval_acts = {l: mg[l] for l in langs}
        eval_note = f"MGSM ({mg[langs[0]].shape[0]} problems, 0-shot)"

    layers_env = os.environ.get("E3R_LAYERS", "all").strip()
    layers = (list(range(n_layers)) if layers_env in ("", "all")
              else sorted({int(x) for x in layers_env.split(",") if x.strip()}))

    fit_all = int(len(fit_sel)) == n_samples          # skip the gather when fitting on everything
    pairs = ordered_or_unordered_pairs(langs, ordered=False)
    print(f"           mode={mode} | eval={eval_note} | {len(pairs)} pairs | ranks={ranks} "
          f"| layers={len(layers)}", flush=True)

    want_pairs = os.environ.get("E3R_PAIRS", "1").strip().lower() not in ("0", "false", "no")
    pivots = [p for p in os.environ.get("E3R_PIVOTS", "en,zh").split(",") if p.strip() in langs]

    rows: list[dict] = []
    pair_rows: list[dict] = []
    for li, k in enumerate(layers):
        t_layer = time.time()
        Afit = ({l: acts[l][:, k, :].float() for l in langs} if fit_all
                else {l: acts[l][fit_sel, k, :].float() for l in langs})
        Aev = {l: eval_acts[l][:, k, :].float() for l in langs}
        W = language_means(Afit)                            # (d, L)

        # Nested bases, fit once at the maximum rank.
        M_nested: dict[str, torch.Tensor] = {
            "diffpca": diffpca_subspace(Afit, fit_rank, angular=True)[0],
            "pooled": pooled_pca_subspace(Afit, fit_rank),
            "random": random_orthonormal_subspace(d, fit_rank, seed),
        }
        # Non-nested bases, refit per rank (cheap: SVD of a (d, L) matrix).
        M_refit: dict[tuple[str, int], torch.Tensor] = {}
        for r in [r for r in ranks if r <= cap]:
            M_refit[("alg1", r)] = build_language_subspace(W, r)[1]
            M_refit[("none", r)] = build_subspace_variant(W, r, mode="none")[0]

        # Accumulate per pair, then average — keeps peak memory at one pair's vectors.
        acc: dict[tuple[str, int], list[float]] = {}
        acc_mean: dict[tuple[str, int], list[float]] = {}
        for (s, t) in pairs:
            V = Aev[t] - Aev[s]                             # (m, d) evaluation translation vectors
            v_bar = (Afit[t].mean(0) - Afit[s].mean(0)).unsqueeze(0)   # (1, d) fit-corpus mean
            # (rho_perprompt, rho_meanvec) for THIS pair, kept only long enough to emit a row.
            per: dict[tuple[str, int], tuple[float, float]] = {}
            for name in NESTED:
                pp = _rho_at_ranks(V, M_nested[name], ranks)
                mv = _rho_at_ranks(v_bar, M_nested[name], ranks)
                for r, val in pp.items():
                    acc.setdefault((name, r), []).append(val)
                    per[(name, r)] = (val, mv[r])
                for r, val in mv.items():
                    acc_mean.setdefault((name, r), []).append(val)
            for (name, r), M in M_refit.items():
                pp_v, mv_v = _rho_single(V, M), _rho_single(v_bar, M)
                acc.setdefault((name, r), []).append(pp_v)
                acc_mean.setdefault((name, r), []).append(mv_v)
                per[(name, r)] = (pp_v, mv_v)
            if want_pairs:
                for (name, r), (pp_v, mv_v) in per.items():
                    pair_rows.append(dict(layer=int(k), subspace=name, rank=int(r),
                                          lang_a=s, lang_b=t, pair=f"{s}|{t}",
                                          rho_perprompt=round(pp_v, 6),
                                          rho_meanvec=round(mv_v, 6)))

        for (name, r), vals in sorted(acc.items()):
            rows.append(dict(layer=int(k), subspace=name, rank=int(r),
                             rho_perprompt=round(float(np.mean(vals)), 6),
                             rho_meanvec=round(float(np.mean(acc_mean[(name, r)])), 6),
                             rank_over_d=round(r / d, 6)))
        print(f"[{time.time()-t_start:7.1f}s] layer {k:3d} ({li+1}/{len(layers)}) "
              f"done in {time.time()-t_layer:5.1f}s", flush=True)

    df = pd.DataFrame(rows)
    # The floor is the MEASURED random subspace at the same layer and rank, not the analytic r/d —
    # they agree closely, but the measured one absorbs any anisotropy in the evaluation vectors.
    floor = (df[df.subspace == "random"].set_index(["layer", "rank"]).rho_perprompt
             .rename("rho_floor"))
    df = df.join(floor, on=["layer", "rank"])
    df["rho_excess"] = (df.rho_perprompt - df.rho_floor).round(6)
    df["mode"] = mode
    df["eval_source"] = eval_note
    df["n_fit_problems"] = int(len(fit_sel))

    out = results_path(f"E3_rank_sweep_{mode}_{tag}_n{cfg.n_samples}.csv", model_tag=tag)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)", flush=True)

    # The per-pair side file. Same floor convention as above, but joined per PAIR — the measured
    # random value on that pair's own vectors, which is the right null for a per-pair comparison.
    pdf = None
    if want_pairs and pair_rows:
        pdf = pd.DataFrame(pair_rows)
        pfloor = (pdf[pdf.subspace == "random"].set_index(["layer", "rank", "pair"]).rho_perprompt
                  .rename("rho_floor"))
        pdf = pdf.join(pfloor, on=["layer", "rank", "pair"])
        pdf["rho_excess"] = (pdf.rho_perprompt - pdf.rho_floor).round(6)
        pdf["mode"] = mode
        pdf["eval_source"] = eval_note
        pout = results_path(f"E3_pairrho_{mode}_{tag}_n{cfg.n_samples}.csv", model_tag=tag)
        pdf.to_csv(pout, index=False)
        print(f"wrote {pout}  ({len(pdf)} rows, {len(pairs)} pairs)", flush=True)

    # ---- read-out ------------------------------------------------------------------------------
    a10 = df[(df.subspace == "alg1") & (df["rank"] == 10)]
    trough_layer = None
    if not a10.empty:
        # Interior-guarded, not a global argmin: an early-layer dip is an embedding-adjacent
        # artifact, not the mid-band subspace. See src/trough.py.
        trough_layer, tv, g_layer, g_value = interior_trough(a10.layer.tolist(),
                                                             a10.rho_perprompt.tolist())
        print(f"\nalg1 r=10 trough: "
              f"{describe(trough_layer, tv, g_layer, g_value, a10.layer.tolist())}")
        if mode == "math":
            print("  gate: compare against the committed E3 headline CSV AT THE SAME n — "
                  "results/E3/<tag>/E3_headline_by_layer_<tag>[_n<N>].csv. The n matters: 7B is "
                  "0.3368 @ L18 at n=512 but 0.4311 @ L18 at n=7500, and E3_results.md quotes the "
                  "n=512 figure. A mismatch at the right n means the protocol drifted; stop here.")
    dp = df[df.subspace == "diffpca"]
    if not dp.empty:
        lay = int(trough_layer) if trough_layer is not None else layers[len(layers)//2]
        at = dp[dp.layer == lay].set_index("rank")
        print(f"\ndiffpca at the trough layer ({lay}) — how much each rank buys:")
        print(f"  {'rank':>5}  {'rho':>7}  {'floor':>7}  {'excess':>7}")
        for r in ranks:
            if r in at.index:
                row = at.loc[r]
                print(f"  {r:5d}  {row.rho_perprompt:7.4f}  {row.rho_floor:7.4f}  {row.rho_excess:7.4f}")
    # The pivot test. At the trough, does rho depend on WHICH languages the pair touches? A pair
    # is "pivot" if either side is in E3R_PIVOTS (en,zh by default). If the pivot rows sit well
    # above the non-pivot rows at the trough and the gap closes by the last layer, that is the
    # English/Chinese-pivot reading with a number attached rather than a colour read off a heatmap.
    if pdf is not None and pivots and not a10.empty:
        lay = int(trough_layer)
        at = pdf[(pdf.subspace == "alg1") & (pdf["rank"] == 10)]
        is_piv = at.lang_a.isin(pivots) | at.lang_b.isin(pivots)
        print(f"\npivot test at the trough (alg1 r=10, layer {lay}) — pivots={','.join(pivots)}:")
        print(f"  {'layer':>5}  {'pivot-pairs':>11}  {'other-pairs':>11}  {'gap':>7}  {'n_piv':>5}")
        for L in sorted({lay, int(df.layer.max())}):
            g = at[at.layer == L]
            p_, o_ = g[is_piv[g.index]].rho_perprompt, g[~is_piv[g.index]].rho_perprompt
            print(f"  {L:5d}  {p_.mean():11.4f}  {o_.mean():11.4f}  "
                  f"{p_.mean()-o_.mean():+7.4f}  {len(p_):5d}")
        print("  a gap that is large at the trough and small at the last layer supports the pivot "
              "reading; a flat gap does not. Per-language detail is in the side file.")

    print(f"\ntotal {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
