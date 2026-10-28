#!/usr/bin/env python
"""Gate the vLLM backend: per-item agreement with the HF path, clean AND intervened.

Runs as THREE steps so each engine uses its own venv — vLLM 0.7.3 pins an older transformers
than the main venv, and importing the HF path under it would fail on unrelated API drift, which
would look like a validation failure instead of a packaging artifact:

  1. `--emit hf   --spec-out S --out A`   (main .venv)      builds+freezes the spec, runs HF
  2. `--emit vllm --spec-in  S --out B`   (.venv-vllm)      loads the SAME spec, runs vLLM
  3. `--compare A B`                       (either)          agreement report + verdict

Freezing the spec to disk (rather than rebuilding it per process) is load-bearing: the two venvs
have different torch builds, so an independently-recomputed SVD could differ in the last bits and
silently weaken the comparison. Both engines must ablate with byte-identical `M_s` and λ.

Two arms, because they fail differently:
  clean       — engine drift (templating, EOS, tokenization, sampling).
  intervened  — the residual-stream trap: vLLM defers the residual add, so a hook editing
                `output[0]` alone perturbs the branch, not the stream. A clean-only check would
                pass while every intervention number was quietly wrong.

Verdict PASS iff both arms agree on >= --min-agree of items, |Δacc| <= --max-acc-delta, and the
vLLM intervened arm actually differs from vLLM clean (else the hooks never fired).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))


def _items(cfg, langs, n):
    from benchmarks import load_benchmark
    out = []
    for lang in langs:
        out += load_benchmark("mgsm", lang, "test", cfg=cfg, n=n, seed=0, n_shot=8)
    return out


def emit_hf(args, cfg, items):
    """Build + freeze the spec, then run both arms on the HF engine."""
    import torch
    from candidates import build_layer_subspaces
    from config import get_device
    from evaluate import run_eval
    from extract import load_model_and_tokenizer, load_or_extract_acts
    from intervene import InterventionSpec, MaskHolder, build_layer_hooks
    from tuning import make_schedule, place_bands, trough_layer

    tag = args.model.split("/")[-1]
    acts = load_or_extract_acts(cfg)
    n_layers = int(next(iter(acts.values())).shape[1])
    subs = build_layer_subspaces(acts, range(n_layers), name="diffpca", rank=cfg.rank,
                                 diffpca_angular=True)
    mid_start, mid_end, high_end = place_bands(trough_layer(tag, n_layers), n_layers)
    lam = make_schedule(mid_start, mid_end, args.lam_mid, high_end, args.lam_high, n_layers)
    torch.save({"subspaces": {int(k): v.cpu() for k, v in subs.items()},
                "lam": {int(k): float(v) for k, v in lam.items()},
                "scope": "prompt_only", "n_layers": n_layers}, args.spec_out)
    print(f"[hf] froze spec -> {args.spec_out} | bands mid[{mid_start},{mid_end}) high->{high_end} "
          f"| {len(lam)} edited layers", flush=True)

    spec = InterventionSpec(subspaces=subs, lam=lam, scope="prompt_only")
    device = get_device()
    model, tok = load_model_and_tokenizer(cfg, device)
    frames = []
    for arm, sp in (("clean", None), ("intervened", spec)):
        mh = MaskHolder(scope="prompt_only")
        hooks = build_layer_hooks(sp, mask_holder=mh) if sp is not None else None
        df = run_eval(model, tok, items, hooks=hooks, mask_holder=(mh if sp else None),
                      device=device, model_name=args.model, cond=arm,
                      max_new_tokens=args.max_new_tokens)
        print(f"[hf] {arm:11s} acc={df['correct'].mean():.4f}", flush=True)
        frames.append(df)
    import pandas as pd
    pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
    print(f"[hf] wrote {args.out}", flush=True)
    return 0


def emit_vllm(args, cfg, items):
    """Load the frozen spec and run both arms on the vLLM engine."""
    import pandas as pd
    import torch
    from intervene import InterventionSpec
    from transformers import AutoTokenizer
    from vllm_backend import VllmRunner, run_eval_vllm

    blob = torch.load(args.spec_in, weights_only=False)
    spec = InterventionSpec(subspaces=blob["subspaces"], lam=blob["lam"], scope=blob["scope"])
    print(f"[vllm] loaded frozen spec: {len(blob['lam'])} edited layers "
          f"of {blob['n_layers']}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    runner = VllmRunner(args.model, max_model_len=max(2048, args.max_new_tokens * 4))
    print(f"[vllm] engine up | {runner.n_layers} decoder layers "
          f"({type(runner.inner).__name__})", flush=True)
    frames = []
    for arm, sp in (("clean", None), ("intervened", spec)):
        df = run_eval_vllm(runner, tok, items, spec=sp, cond=arm,
                           max_new_tokens=args.max_new_tokens, model_name=args.model)
        print(f"[vllm] {arm:11s} acc={df['correct'].mean():.4f}", flush=True)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
    print(f"[vllm] wrote {args.out}", flush=True)
    return 0


def compare(args):
    import pandas as pd
    hf = pd.read_csv(args.compare[0])
    vl = pd.read_csv(args.compare[1])
    rows, ok = [], True
    for arm in ("clean", "intervened"):
        a = hf[hf["cond"] == arm].set_index("id")
        b = vl[vl["cond"] == arm].set_index("id")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            print(f"[FAIL] no shared items for arm {arm}")
            return 1
        # `.astype(str)` alone leaves a missing answer as float nan, and nan != nan -- so two
        # engines that BOTH failed to parse an answer scored as disagreeing. On a floored language
        # that is most of the items. Fold missing to one sentinel first.
        _pa = a.loc[common, "pred"].astype("string").fillna("<none>").astype(str)
        _pb = b.loc[common, "pred"].astype("string").fillna("<none>").astype(str)
        agree_pred = float((_pa.values == _pb.values).mean())
        d_acc = float(b.loc[common, "correct"].mean() - a.loc[common, "correct"].mean())
        arm_ok = (agree_pred >= args.min_agree) and (abs(d_acc) <= args.max_acc_delta)
        ok &= arm_ok
        rows.append({"arm": arm, "n": len(common), "agree_pred": round(agree_pred, 4),
                     "acc_hf": round(float(a.loc[common, "correct"].mean()), 4),
                     "acc_vllm": round(float(b.loc[common, "correct"].mean()), 4),
                     "delta_acc": round(d_acc, 4), "pass": arm_ok})
    rep = pd.DataFrame(rows)
    print("\n" + rep.to_string(index=False), flush=True)

    # the hooks must actually bite: if vLLM's intervened arm equals its clean arm, agreement is
    # vacuous (we'd be comparing two clean runs and calling the intervention validated)
    vc = vl[vl["cond"] == "clean"].set_index("id")["pred"].astype(str)
    vi = vl[vl["cond"] == "intervened"].set_index("id")["pred"].astype(str)
    shared = vc.index.intersection(vi.index)
    changed = float((vc.loc[shared] != vi.loc[shared]).mean())
    hc = hf[hf["cond"] == "clean"].set_index("id")["pred"].astype(str)
    hi = hf[hf["cond"] == "intervened"].set_index("id")["pred"].astype(str)
    hshared = hc.index.intersection(hi.index)
    hchanged = float((hc.loc[hshared] != hi.loc[hshared]).mean())
    print(f"\n[sanity] intervention changed predictions on: HF {hchanged:.1%} | vLLM {changed:.1%}",
          flush=True)
    if changed == 0.0:
        print("[FAIL] vLLM hooks were a no-op — the edit never reached the residual stream")
        ok = False

    out = _root / "results" / "vllm_validation_report.csv"
    out.parent.mkdir(exist_ok=True)
    rep.to_csv(out, index=False)
    print(f"\nwrote {out}\nVERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=("hf", "vllm"))
    ap.add_argument("--compare", nargs=2, metavar=("HF_CSV", "VLLM_CSV"))
    ap.add_argument("--model", default=os.environ.get("E5_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--langs", default="en,sw")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lam-mid", type=float, default=0.4)
    ap.add_argument("--lam-high", type=float, default=-0.2)
    ap.add_argument("--min-agree", type=float, default=0.95)
    ap.add_argument("--max-acc-delta", type=float, default=0.02)
    ap.add_argument("--n-samples", type=int, default=int(os.environ.get("E5_N_SAMPLES", "7500")))
    ap.add_argument("--spec-out", default="results/vllm_val_spec.pt")
    ap.add_argument("--spec-in", default="results/vllm_val_spec.pt")
    ap.add_argument("--out", default="results/vllm_val_arm.csv")
    args = ap.parse_args()

    if args.compare:
        return compare(args)

    from config import Config, set_seeds
    set_seeds(0)
    cfg = Config(model_name=args.model, n_samples=args.n_samples)
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    items = _items(cfg, langs, args.n)
    print(f"[{args.emit}] {len(items)} items across {langs}", flush=True)
    return emit_hf(args, cfg, items) if args.emit == "hf" else emit_vllm(args, cfg, items)


if __name__ == "__main__":
    raise SystemExit(main())
