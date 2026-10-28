#!/usr/bin/env python
"""Gate the INTEGRATED E5_ENGINE routing: run one real dev eval through `evaluate.run_eval` and
`tuning.coordinate_search` on the HF engine (E5_ENGINE=hf) vs the vLLM engine (E5_ENGINE=vllm) and
prove (a) identical DataFrame schema, (b) accuracy agreement within a few points, (c) a real
wall-clock speedup — the whole point of the vLLM backend.

Unlike scripts/validate_vllm_vs_hf.py (which calls `run_eval_vllm` directly), this drives the
PRODUCTION seam: `run_eval` itself dispatches to the module-level vLLM singleton when E5_ENGINE=vllm,
and `coordinate_search` reaches it unchanged through `_eval_schedule_df`. So this also validates
(d) the singleton is built ONCE and reused across the search's many evals.

Two engines run in their OWN venv (vLLM 0.7.3 pins transformers 4.49; the HF venv is newer), so the
spec is frozen to disk by the HF step and loaded verbatim by the vLLM step — both ablate with
byte-identical M_s and λ (an independently recomputed SVD could differ in the last bits).

  1. --emit hf   --spec-out S --out-csv A --out-json JA   (.venv)       freeze spec, HF arms + timing
  2. --emit vllm --spec-in  S --out-csv B --out-json JB   (vllm-lrd)    same evals on vLLM + timing
  3. --compare A B JA JB                                   (.venv)       schema + accuracy + speedup
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))


def _items(cfg, lang, n):
    from benchmarks import load_benchmark
    return load_benchmark("mgsm", lang, "test", cfg=cfg, n=n, seed=0, n_shot=8)


def _spec_from(subs, lam, scope="prompt_only"):
    from intervene import InterventionSpec
    return InterventionSpec(subspaces=subs, lam=lam, scope=scope)


def emit_hf(args, cfg):
    """Freeze the DiffPCA spec, then run clean+intervened + a short search on the HF engine."""
    import torch
    from candidates import build_layer_subspaces
    from config import get_device
    from evaluate import run_eval
    from extract import load_model_and_tokenizer, load_or_extract_acts
    from intervene import MaskHolder, build_layer_hooks
    from tuning import coordinate_search, make_schedule, place_bands, trough_layer

    tag = args.model.split("/")[-1]
    device = get_device()
    model, tok = load_model_and_tokenizer(cfg, device)
    acts = load_or_extract_acts(cfg)
    n_layers = int(next(iter(acts.values())).shape[1])
    subs = build_layer_subspaces(acts, range(n_layers), name="diffpca", rank=cfg.rank,
                                 diffpca_angular=True)
    ms, me, he = place_bands(trough_layer(tag, n_layers), n_layers)
    lam = make_schedule(ms, me, args.lam_mid, he, args.lam_high, n_layers)
    torch.save({"subspaces": {int(k): v.cpu() for k, v in subs.items()},
                "lam": {int(k): float(v) for k, v in lam.items()},
                "scope": "prompt_only", "n_layers": n_layers}, args.spec_out)
    spec = _spec_from(subs, lam)
    print(f"[hf] froze spec -> {args.spec_out} | {len(lam)} edited layers of {n_layers}", flush=True)
    return _run_arm("hf", args, cfg, tok, model, device, spec, subs, n_layers)


def emit_vllm(args, cfg):
    """Load the frozen spec and run the SAME evals through the integrated vLLM route (E5_ENGINE=vllm).

    No HF model is loaded here — on the vLLM path run_eval ignores it, so we free that VRAM for the
    engine. The tokenizer + frozen subspaces/λ are all the intervention needs.
    """
    import torch
    from transformers import AutoTokenizer

    from corpus import build_chat_inputs

    blob = torch.load(args.spec_in, weights_only=False)
    subs = {int(k): v for k, v in blob["subspaces"].items()}
    spec = _spec_from(subs, {int(k): float(v) for k, v in blob["lam"].items()}, blob["scope"])
    n_layers = int(blob["n_layers"])
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[vllm] loaded frozen spec: {len(blob['lam'])} edited layers of {n_layers}", flush=True)

    # Size the KV-cache window from the ACTUAL prompts so a floored language (bn: ~2k-token 8-shot
    # prompts) still has room to generate — a fixed window silently truncated bn to 0 tokens once.
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    longest = 0
    for lang in langs:
        P = build_chat_inputs(tok, [it.prompt for it in _items(cfg, lang, args.n)])
        longest = max(longest, max(len(x) for x in tok(P)["input_ids"]))
    os.environ["E5_VLLM_MAX_MODEL_LEN"] = str(longest + args.max_new_tokens + 128)
    print(f"[vllm] longest prompt {longest} tok -> E5_VLLM_MAX_MODEL_LEN "
          f"{os.environ['E5_VLLM_MAX_MODEL_LEN']}", flush=True)
    return _run_arm("vllm", args, cfg, tok, None, None, spec, subs, n_layers)


def _run_arm(engine, args, cfg, tok, model, device, spec, subs, n_layers):
    """Shared body: clean+intervened run_eval per lang (for schema/accuracy) + one short
    coordinate_search (for the search-path timing and singleton-reuse proof). E5_ENGINE selects the
    engine INSIDE run_eval — this function is engine-agnostic by design."""
    import pandas as pd
    import evaluate as _ev
    from evaluate import run_eval
    from intervene import MaskHolder, build_layer_hooks
    from tuning import coordinate_search

    os.environ["E5_ENGINE"] = engine
    tag = args.model.split("/")[-1]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    def _eval(items, sp, cond):
        mh = MaskHolder(scope="prompt_only")
        hooks = build_layer_hooks(sp, mask_holder=mh) if sp is not None else None
        return run_eval(model, tok, items, hooks=hooks, mask_holder=(mh if sp else None),
                        device=device, model_name=args.model, cond=cond,
                        max_new_tokens=args.max_new_tokens, intervention_spec=sp)

    # warm-up (build the vLLM engine / warm HF kernels) — timing discarded
    _ = _eval(_items(cfg, langs[0], min(8, args.n)), None, "warmup")
    if engine == "vllm":
        assert len(_ev._VLLM_RUNNERS) == 1, f"expected 1 warm engine, got {len(_ev._VLLM_RUNNERS)}"

    frames, timing = [], {}
    for lang in langs:
        items = _items(cfg, lang, args.n)
        df_clean = _eval(items, None, "clean")
        t0 = time.perf_counter()
        df_intv = _eval(items, spec, "intervention")
        dt = time.perf_counter() - t0
        frames += [df_clean, df_intv]
        toks = int(df_intv["n_new_tokens"].sum())
        timing[lang] = {"seconds": round(dt, 2), "n": len(items), "gen_tokens": toks,
                        "tok_per_s": round(toks / dt, 1) if dt > 0 else 0.0,
                        "trunc_rate": round(float(df_intv["truncated"].mean()), 3),
                        "acc_clean": round(float(df_clean["correct"].mean()), 4),
                        "acc_intv": round(float(df_intv["correct"].mean()), 4)}
        print(f"[{engine}] {lang}: intervened {dt:.1f}s  {toks} tok  "
              f"{timing[lang]['tok_per_s']} tok/s  acc {timing[lang]['acc_intv']}  "
              f"trunc {timing[lang]['trunc_rate']}", flush=True)
        if engine == "vllm" and toks == 0:
            print(f"[FAIL] vLLM generated 0 tokens for {lang} — window too small", flush=True)
            return 1

    # --- search path: a short coordinate_search reaches run_eval through _eval_schedule_df, so this
    # times the unit a phase actually runs and proves the singleton is reused across every trial.
    dev = _items(cfg, langs[-1], args.n)                 # the floored lang (bn) — where vLLM wins most
    grids = {"lam_mid": (0.2, 0.4), "lam_high": (0.0, -0.2)}
    t0 = time.perf_counter()
    # NB no `label=` here: coordinate_search takes no such kwarg and has no **kwargs, so passing
    # one raised TypeError *after* the model load, the engine build and every per-language eval.
    _spec, log = coordinate_search(model, tok, dev, subs, n_layers, model_tag=tag,
                                   grids=grids, device=device, model_name=args.model)
    dt = time.perf_counter() - t0
    n_evals = len(log)
    timing["_search"] = {"seconds": round(dt, 2), "n_evals": int(n_evals),
                         "best_acc": round(float(log["acc"].max()), 4),
                         "dev_n": len(dev), "lang": langs[-1]}
    if engine == "vllm":
        timing["_search"]["n_engines_built"] = len(_ev._VLLM_RUNNERS)   # must stay 1
    print(f"[{engine}] coordinate_search: {dt:.1f}s over {n_evals} evals "
          f"(best_acc {timing['_search']['best_acc']}"
          + (f", engines_built {len(_ev._VLLM_RUNNERS)}" if engine == "vllm" else "") + ")",
          flush=True)

    pd.concat(frames, ignore_index=True).to_csv(args.out_csv, index=False)
    pathlib.Path(args.out_json).write_text(json.dumps(timing, indent=2))
    print(f"[{engine}] wrote {args.out_csv} + {args.out_json}", flush=True)
    return 0


def compare(args):
    import pandas as pd
    hf = pd.read_csv(args.compare[0])
    vl = pd.read_csv(args.compare[1])
    jh = json.loads(pathlib.Path(args.compare[2]).read_text())
    jv = json.loads(pathlib.Path(args.compare[3]).read_text())
    ok = True

    # (a) schema — identical columns AND order (MGSM-only => vLLM matches _run_generative exactly)
    same_set = set(hf.columns) == set(vl.columns)
    same_order = list(hf.columns) == list(vl.columns)
    print(f"[schema] columns identical set={same_set} order={same_order}")
    if not same_set:
        print(f"  HF only : {sorted(set(hf.columns) - set(vl.columns))}")
        print(f"  vLLM only: {sorted(set(vl.columns) - set(hf.columns))}")
    ok &= same_set

    # (b) accuracy agreement per (lang, cond) within tolerance
    print(f"\n[accuracy] tolerance +/- {args.max_acc_delta}")
    print(f"  {'lang':6s} {'cond':13s} {'n':>4s} {'acc_hf':>7s} {'acc_vllm':>9s} {'|d|':>6s} pass")
    for (lang, cond), a in hf.groupby(["lang", "cond"]):
        b = vl[(vl["lang"] == lang) & (vl["cond"] == cond)]
        if len(b) == 0:
            print(f"  {lang:6s} {cond:13s} MISSING in vLLM"); ok = False; continue
        da = abs(float(a["correct"].mean()) - float(b["correct"].mean()))
        arm_ok = da <= args.max_acc_delta
        ok &= arm_ok
        print(f"  {lang:6s} {cond:13s} {len(a):4d} {a['correct'].mean():7.3f} "
              f"{b['correct'].mean():9.3f} {da:6.3f} {'Y' if arm_ok else 'N'}")

    # sanity: the intervention must actually bite on BOTH engines (else agreement is vacuous)
    for eng, df in (("hf", hf), ("vllm", vl)):
        c = df[df["cond"] == "clean"].set_index("id")["pred"].astype(str)
        i = df[df["cond"] == "intervention"].set_index("id")["pred"].astype(str)
        sh = c.index.intersection(i.index)
        chg = float((c.loc[sh] != i.loc[sh]).mean()) if len(sh) else 0.0
        print(f"[sanity] {eng}: intervention changed {chg:.1%} of predictions")
        if eng == "vllm" and chg == 0.0:
            print("[FAIL] vLLM hooks were a no-op"); ok = False

    # (c) speedup — per lang (intervened run_eval) + the search path (the phase unit)
    print("\n[speedup]")
    print(f"  {'unit':10s} {'HF s':>9s} {'vLLM s':>9s} {'speedup':>8s} {'HF tok/s':>9s} {'vLLM tok/s':>11s}")
    tot_h = tot_v = 0.0
    for lang in [k for k in jh if not k.startswith("_")]:
        if lang not in jv:
            continue
        h, v = jh[lang], jv[lang]
        # only a valid comparison if both engines did comparable work
        if v["gen_tokens"] < 0.5 * h["gen_tokens"]:
            print(f"  [INVALID] {lang}: vLLM {v['gen_tokens']} tok vs HF {h['gen_tokens']} tok"); ok = False
        tot_h += h["seconds"]; tot_v += v["seconds"]
        print(f"  {lang:10s} {h['seconds']:9.1f} {v['seconds']:9.1f} "
              f"{h['seconds']/max(v['seconds'],1e-9):7.2f}x {h['tok_per_s']:9.1f} {v['tok_per_s']:11.1f}")
    if "_search" in jh and "_search" in jv:
        h, v = jh["_search"], jv["_search"]
        print(f"  {'search':10s} {h['seconds']:9.1f} {v['seconds']:9.1f} "
              f"{h['seconds']/max(v['seconds'],1e-9):7.2f}x   (over {h['n_evals']} evals; "
              f"vLLM engines_built={v.get('n_engines_built')})")
        if v.get("n_engines_built") not in (None, 1):
            print(f"[FAIL] vLLM built {v['n_engines_built']} engines — singleton not reused"); ok = False
    print(f"\n  blended per-lang: HF {tot_h:.1f}s vs vLLM {tot_v:.1f}s -> "
          f"{tot_h/max(tot_v,1e-9):.2f}x")

    out = _root / "results" / f"engine_parity_report_{args.model.split('/')[-1]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"hf_timing": jh, "vllm_timing": jv,
                               "blended_speedup": round(tot_h/max(tot_v,1e-9), 3),
                               "verdict": "PASS" if ok else "FAIL"}, indent=2))
    print(f"\nwrote {out}\nVERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=("hf", "vllm"))
    ap.add_argument("--compare", nargs=4, metavar=("HF_CSV", "VLLM_CSV", "HF_JSON", "VLLM_JSON"))
    ap.add_argument("--model", default=os.environ.get("E5_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    ap.add_argument("--n", type=int, default=125)
    ap.add_argument("--langs", default="en,bn")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lam-mid", type=float, default=0.4)
    ap.add_argument("--lam-high", type=float, default=-0.2)
    ap.add_argument("--max-acc-delta", type=float, default=0.06)
    ap.add_argument("--n-samples", type=int, default=int(os.environ.get("E5_N_SAMPLES", "7500")))
    ap.add_argument("--spec-out", default="results/engine_parity_spec.pt")
    ap.add_argument("--spec-in", default="results/engine_parity_spec.pt")
    ap.add_argument("--out-csv", default="results/engine_parity_arm.csv")
    ap.add_argument("--out-json", default="results/engine_parity_arm.json")
    args = ap.parse_args()

    if args.compare:
        return compare(args)
    from config import Config, set_seeds
    set_seeds(0)
    cfg = Config(model_name=args.model, n_samples=args.n_samples)
    return emit_hf(args, cfg) if args.emit == "hf" else emit_vllm(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
