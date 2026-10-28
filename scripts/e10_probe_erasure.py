"""E10 — does ablation erase language from the residual stream, or only move it?

Design and rationale: docs/experiments/E10_probe_erasure.md. Two probes on the same edited
activations, and the GAP between them is the result:

  frozen     trained on CLEAN activations, evaluated on ABLATED ones. Asks whether the edit
             pushed activations off the clean decision boundary. On its own it is not evidence of
             erasure -- ANY large perturbation, a random subspace included, defeats it.
  retrained  trained AND evaluated on ABLATED activations. Asks whether language is still linearly
             decodable at all. This is the one LEACE would trivially win, which is why LEACE and
             Wiener are excluded here and only orthogonal projection is swept.

Frozen at the floor while retrained recovers = the signal was displaced, not erased.

## Corpora — one transfer, not two

The subspace is fit on the MATH activation cache; the probe trains and evaluates on MGSM, grouped
by problem index so the parallel translations of one problem never straddle a fold (they share
content, and an ungrouped split lets the probe memorize problems instead of learning language).
Were the probe also trained on MATH, subspace transfer and probe transfer would be confounded and
a drop could not be attributed to either.

## Slices, not a factorial

The full cross of layers x subspaces x ranks x lambdas is ~4,500 cells, each a 5-fold multinomial
fit on (2750 x d) -- days of CPU for a figure that needs three lines. This runs the three slices
the figures actually consume, and nothing else:

  layer  all layers x {alg1, diffpca} at rank 10, lam 1        -> Figure 1 (frozen vs retrained)
  rank   trough layer x all subspaces x the rank ladder, lam 1 -> Figure 2 (how many dimensions)
  dose   trough layer x all subspaces at rank 10 x the lambdas -> Figure 3 (how hard a push)

Clean (unedited) accuracy is measured once per layer touched, and the label-permutation floor once
per layer, since neither depends on the arm.

Environment knobs (set by scripts/prajna/e10_prajna.sbatch; the job is --export=NONE):

  LRD_MODEL        HF model id
  LRD_N_SAMPLES    MATH construction-corpus size            (default 7500)
  E10_SLICES       comma-separated: layer,rank,dose          (default all three)
  E10_RANKS        rank ladder for the `rank` slice          (default 1,2,3,5,10,20,50,100)
  E10_LAMS         lambda ladder for the `dose` slice        (default 0,0.25,0.5,0.75,1.0)
  E10_TROUGH       trough layer; "" -> read from results/shared/trough_<tag>.json
  E10_LAYERS       layers for the `layer` slice, or "all"    (default all)
  E10_FOLDS        GroupKFold splits                         (default 5)
  E10_PERMS        label permutations for the floor          (default 5)
  E10_FLOOR_LAYERS layers to MEASURE the floor at, or "all"   (default: first, trough, last)
  E10_MGSM_ACTS    MGSM activation cache path

Usage:
    python scripts/e10_probe_erasure.py
"""
from __future__ import annotations

import json
import os
import re
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
             if (p / "src" / "config.py").exists())
sys.path.insert(0, str(_root / "src"))

from candidates import diffpca_subspace                                  # noqa: E402
from config import ACTS_DIR, Config, RESULTS_DIR, results_path, set_seeds  # noqa: E402
from controls import pooled_pca_subspace, random_orthonormal_subspace    # noqa: E402
from extract import acts_cache_path                                      # noqa: E402
from intervene import ablate                                             # noqa: E402
from subspace import build_language_subspace, language_means             # noqa: E402

SUBSPACES = ("alg1", "diffpca", "pooled", "random")
CAPPED = ("alg1",)            # structurally limited to L-1; `none` is the same span, so it is out


def _probe(seed: int):
    """The probe. Same settings as `probe.train_language_probe`, so numbers stay comparable."""
    return LogisticRegression(max_iter=3000, C=1.0, random_state=seed)


def _cv_accuracy(X_train, X_eval, y, groups, n_splits: int, seed: int) -> float:
    """Grouped-CV accuracy, training on `X_train` and testing on `X_eval`.

    Passing the SAME array twice gives the retrained probe; passing (clean, edited) gives the
    frozen probe. Doing frozen fold-wise rather than as one global fit keeps the group discipline
    identical across the two, so the gap between them is not an artefact of different splits.
    """
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X_train, y, groups):
        clf = _probe(seed).fit(X_train[tr], y[tr])
        accs.append(float((clf.predict(X_eval[te]) == y[te]).mean()))
    return float(np.mean(accs))


def _floor(X, y, groups, n_splits: int, n_perms: int, seed: int) -> float:
    """Empirical chance level by label permutation — the honest baseline, not the analytic 1/L.

    Measured at a few layers rather than all of them (`E10_FLOOR_LAYERS`). Under permuted labels
    the probe is fitting noise, so the value is ~1/L wherever it is taken; measuring it 28 times
    cost ~70% of the job's wall-clock for a quantity that does not move. Layers that are not
    measured inherit the mean of the measured ones and are flagged `floor_measured=False`.
    """
    rng = np.random.default_rng(seed)
    return float(np.mean([_cv_accuracy(X, X, rng.permutation(y), groups, n_splits, seed)
                          for _ in range(n_perms)]))


def _build(name: str, Afit: dict, W: torch.Tensor, rank: int, d: int, seed: int) -> torch.Tensor:
    if name == "alg1":
        return build_language_subspace(W, rank)[1]
    if name == "diffpca":
        return diffpca_subspace(Afit, rank, angular=True)[0]
    if name == "pooled":
        return pooled_pca_subspace(Afit, rank)
    if name == "random":
        return random_orthonormal_subspace(d, rank, seed)
    raise ValueError(name)


_ROW_COLS = ["layer", "subspace", "rank", "lam", "probe_kind", "acc", "acc_floor", "floor_measured"]


def _flush_inflight(path: pathlib.Path, rows: list[dict], trough: int) -> None:
    """Write the rows so far, atomically, stamped with the trough they were computed against."""
    df = pd.DataFrame(rows, columns=_ROW_COLS)
    df["trough_layer"] = trough
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_inflight(path: pathlib.Path, trough: int) -> tuple[list[dict], dict[int, float], set[int]]:
    """Rows, measured floors, and completed layers from a previous run's _INFLIGHT file.

    Empty when there is none, when E10_RESUME=0, or when the file was written against a different
    trough -- the rank and dose ladders exist only at the trough, so such rows cannot be reused.
    """
    if not path.exists() or os.environ.get("E10_RESUME", "1") == "0":
        return [], {}, set()
    df = pd.read_csv(path)
    if "trough_layer" not in df.columns or int(df.trough_layer.iloc[0]) != trough:
        print(f"           WARN: {path.name} was written at trough "
              f"L{df.trough_layer.iloc[0] if 'trough_layer' in df.columns else '?'}, "
              f"this run is L{trough} -- ignoring it", flush=True)
        return [], {}, set()
    df["floor_measured"] = df.floor_measured.astype(bool)
    rows = df[_ROW_COLS].to_dict("records")
    meas = df[(df.subspace == "clean") & df.floor_measured]
    measured = {int(r.layer): float(r.acc_floor) for r in meas.itertuples()}
    return rows, measured, {int(k) for k in df.layer.unique()}


def main() -> int:
    t0 = time.time()
    seed = 0
    set_seeds(seed)
    cfg = Config(model_name=os.environ.get("LRD_MODEL", Config.model_name),
                 n_samples=int(os.environ.get("LRD_N_SAMPLES", "7500")))
    tag = cfg.model_name.split("/")[-1]
    langs = list(cfg.langs)
    L = len(langs)
    cap = L - 1
    folds = int(os.environ.get("E10_FOLDS", "5"))
    perms = int(os.environ.get("E10_PERMS", "5"))
    ranks = sorted({int(r) for r in os.environ.get("E10_RANKS", "1,2,3,5,10,20,50,100").split(",")})
    lams = [float(x) for x in os.environ.get("E10_LAMS", "0,0.25,0.5,0.75,1.0").split(",")]
    slices = {s.strip() for s in os.environ.get("E10_SLICES", "layer,rank,dose").split(",")}

    # ---- fit corpus (MATH) and probe corpus (MGSM) ---------------------------------------------
    cache = acts_cache_path(cfg)
    if not cache.exists():
        print(f"ERROR: MATH activation cache missing: {cache}", file=sys.stderr)
        return 3
    mgsm_path = pathlib.Path(os.environ.get("E10_MGSM_ACTS", ACTS_DIR / f"mgsm_acts_{tag}_nshot0.pt"))
    if not mgsm_path.exists():
        print(f"ERROR: MGSM activation cache missing: {mgsm_path}\n"
              f"       run the E3 'mgsmacts' phase for this model first.", file=sys.stderr)
        return 3
    print(f"[{time.time()-t0:7.1f}s] loading {cache}", flush=True)
    acts = torch.load(cache, map_location="cpu", weights_only=True)
    print(f"[{time.time()-t0:7.1f}s] loading {mgsm_path}", flush=True)
    mg = torch.load(mgsm_path, map_location="cpu", weights_only=True)
    n_fit, n_layers, d = acts[langs[0]].shape
    n_items = mg[langs[0]].shape[0]
    if mg[langs[0]].shape[1:] != (n_layers, d):
        print("ERROR: MGSM acts do not match the fit cache shape — different model?", file=sys.stderr)
        return 3

    # Probe labels and groups are layer-independent: language-major stacking, problem index tiled.
    y = np.repeat(np.arange(L), n_items)
    groups = np.tile(np.arange(n_items), L)

    # ---- trough ---------------------------------------------------------------------------------
    tro_env = os.environ.get("E10_TROUGH", "").strip()
    if tro_env:
        trough = int(tro_env)
    else:
        tj = RESULTS_DIR / "shared" / f"trough_{tag}.json"
        if not tj.exists():
            print(f"ERROR: no trough for {tag} ({tj}); set E10_TROUGH.", file=sys.stderr)
            return 3
        trough = int(json.loads(tj.read_text())["trough_layer"])
    print(f"           tag={tag} d={d} layers={n_layers} trough=L{trough} "
          f"| MGSM {n_items} items x {L} langs | slices={sorted(slices)}", flush=True)

    fl_env = os.environ.get("E10_FLOOR_LAYERS", "").strip()
    layers_env = os.environ.get("E10_LAYERS", "all").strip()
    layer_slice_layers = (list(range(n_layers)) if layers_env in ("", "all")
                          else sorted({int(x) for x in layers_env.split(",") if x.strip()}))

    # (layer, subspace, rank, lam) cells, deduplicated across slices.
    cells: set[tuple[int, str, int, float]] = set()
    if "layer" in slices:
        cells |= {(k, s, 10, 1.0) for k in layer_slice_layers for s in ("alg1", "diffpca")}
    if "rank" in slices:
        cells |= {(trough, s, r, 1.0) for s in SUBSPACES for r in ranks
                  if not (s in CAPPED and r > cap)}
    if "dose" in slices:
        cells |= {(trough, s, 10, lam) for s in SUBSPACES for lam in lams}
    by_layer: dict[int, list[tuple[str, int, float]]] = {}
    for k, s, r, lam in sorted(cells):
        by_layer.setdefault(k, []).append((s, r, lam))
    print(f"           {len(cells)} arm cells over {len(by_layer)} layers "
          f"(~{2*folds*len(cells)} probe fits + baselines)", flush=True)

    touched = sorted(by_layer)
    if fl_env in ("all",):
        floor_layers = set(touched)
    elif fl_env:
        floor_layers = {int(x) for x in fl_env.split(",") if x.strip()} & set(touched)
    else:
        floor_layers = {touched[0], trough, touched[-1]} & set(touched)
    print(f"           floor measured at layers {sorted(floor_layers)} x {perms} perms "
          f"(others inherit the mean — see _floor)", flush=True)

    # ---- per-layer checkpoint --------------------------------------------------------------------
    # The trough layer alone is ~45 arms (3.7 h on an l40; job 301105 lost 14 h to a walltime kill
    # with the layer slice done and the trough block half-way). Rows are flushed to an _INFLIGHT
    # CSV after every layer, and a resubmit skips the layers already there. A layer is either fully
    # present or absent, because the flush happens only at the layer boundary. E10_RESUME=0 ignores
    # the file; a different trough invalidates it (the rank/dose ladders live at the trough).
    # E10_OUT_SUFFIX lets the run be split across concurrent jobs BY SLICE, which is the only
    # safe axis: the layer slice touches every layer so it measures floors at
    # {first, trough, last} exactly as an undivided run would, and the rank/dose slices live
    # wholly at the trough, where the floor is measured directly and nothing is inherited.
    # Splitting by LAYER instead would change which layers the floor is measured at, and so
    # change the mean the remaining layers inherit -- do not do that. Shards are recombined
    # by scripts/glm_merge_probe.py, which de-duplicates the (layer, subspace, rank, lam)
    # cells the two slices legitimately share at the trough.
    suffix = os.environ.get("E10_OUT_SUFFIX", "").strip()
    if suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", suffix):
        raise ValueError("E10_OUT_SUFFIX must be letters, digits, underscore or hyphen")
    out = results_path(f"E10_probe_erasure_{tag}{'_' + suffix if suffix else ''}.csv",
                       model_tag=tag)
    inflight = out.with_name(out.stem + "_INFLIGHT.csv")
    rows, measured, done = _load_inflight(inflight, trough)
    if done:
        print(f"           resuming from {inflight.name}: {len(done)} layers done "
              f"{sorted(done)}, floors measured at {sorted(measured)}", flush=True)

    for ki, (k, arms) in enumerate(sorted(by_layer.items())):
        if k in done:
            print(f"           layer {k} skipped (in _INFLIGHT) ({ki+1}/{len(by_layer)})", flush=True)
            continue
        t_k = time.time()
        Afit = {l: acts[l][:, k, :].float() for l in langs}
        W = language_means(Afit)
        Xc = torch.cat([mg[l][:, k, :].float() for l in langs], dim=0)      # (L*n_items, d) clean
        Xc_np = Xc.numpy()

        clean = _cv_accuracy(Xc_np, Xc_np, y, groups, folds, seed)
        if k in floor_layers:
            fl = measured[k] = _floor(Xc_np, y, groups, folds, perms, seed)
        else:
            fl = float(np.mean(list(measured.values()))) if measured else 1.0 / L
        is_meas = k in floor_layers
        rows.append(dict(layer=k, subspace="clean", rank=0, lam=0.0, probe_kind="retrained",
                         acc=round(clean, 6), acc_floor=round(fl, 6), floor_measured=is_meas))
        print(f"[{time.time()-t0:7.1f}s] L{k:<3d} clean={clean:.4f} "
              f"floor={fl:.4f}{'' if is_meas else ' (inherited)'} | {len(arms)} arms", flush=True)

        # Subspaces are cached per (name, rank) so the dose ladder does not refit them per lambda.
        M_cache: dict[tuple[str, int], torch.Tensor] = {}
        for (name, r, lam) in arms:
            M = M_cache.get((name, r))
            if M is None:
                M = M_cache[(name, r)] = _build(name, Afit, W, r, d, seed)
            Xe_np = ablate(Xc, M, lam).numpy()
            frozen = _cv_accuracy(Xc_np, Xe_np, y, groups, folds, seed)
            retr = _cv_accuracy(Xe_np, Xe_np, y, groups, folds, seed)
            for kind, a in (("frozen", frozen), ("retrained", retr)):
                rows.append(dict(layer=k, subspace=name, rank=r, lam=lam, probe_kind=kind,
                                 acc=round(a, 6), acc_floor=round(fl, 6), floor_measured=is_meas))
            print(f"           L{k:<3d} {name:<8s} r={r:<4d} lam={lam:<5.2f} "
                  f"frozen={frozen:.4f} retrained={retr:.4f} gap={retr-frozen:+.4f}", flush=True)
        print(f"[{time.time()-t0:7.1f}s] layer {k} done in {time.time()-t_k:.1f}s "
              f"({ki+1}/{len(by_layer)})", flush=True)
        _flush_inflight(inflight, rows, trough)

    df = pd.DataFrame(rows)
    if measured:
        # Layers processed before any floor had been measured inherited 1/L; now that every
        # measurement is in, backfill them all from the same mean so the band is consistent.
        mean_floor = round(float(np.mean(list(measured.values()))), 6)
        df.loc[~df.floor_measured, "acc_floor"] = mean_floor
    df["acc_over_floor"] = (df.acc - df.acc_floor).round(6)
    df["model_tag"] = tag
    df["trough_layer"] = trough
    df.to_csv(out, index=False)
    inflight.unlink(missing_ok=True)
    print(f"\nwrote {out}  ({len(df)} rows, {time.time()-t0:.1f}s)")

    head = df[(df.layer == trough) & (df["rank"] == 10) & (df.lam == 1.0)]
    if not head.empty:
        print(f"\nheadline — trough L{trough}, rank 10, lam 1.0:")
        piv = head.pivot_table(index="subspace", columns="probe_kind", values="acc")
        if {"frozen", "retrained"} <= set(piv.columns):
            piv["gap"] = (piv.retrained - piv.frozen).round(4)
        print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
