#!/usr/bin/env python
"""Decisive gate for the vLLM intervention math: first-token agreement, not end-to-end accuracy.

Why not accuracy: greedy decoding is chaotic. One token flipped by a bf16 kernel difference
cascades into a completely different continuation, and on a benchmark where ~2/3 of answers are
wrong, wrong answers are near-arbitrary strings. The end-to-end run therefore showed only ~42%
per-item agreement even on the CLEAN arm, where the two engines run identical mathematics — so it
cannot discriminate "my residual-stream reconstruction is wrong" from "kernels differ in the last
bits". It is the wrong instrument.

What this measures instead: exactly one forward pass per prompt (`max_tokens=1`) and compares the
argmax token plus the top-k logprob vector. With no decode loop there is no chaos to amplify, so:

  * CLEAN arm  — establishes the engines' intrinsic numerical floor (expect very high agreement).
  * INTERVENED — must match the clean arm's agreement. If the residual reconstruction is wrong,
                 the edited stream differs by a large, systematic amount and first-token agreement
                 collapses far below the clean floor. That is the signal we actually need.

The comparison is thus RELATIVE (intervened vs clean floor), which is robust to whatever the
engines' baseline numerical disagreement happens to be.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))


def _prompts(cfg, langs, n, tok):
    from benchmarks import load_benchmark
    from corpus import build_chat_inputs
    items = []
    for lang in langs:
        items += load_benchmark("mgsm", lang, "test", cfg=cfg, n=n, seed=0, n_shot=8)
    return items, build_chat_inputs(tok, [it.prompt for it in items])


def emit_hf(args, cfg):
    import torch
    from candidates import build_layer_subspaces
    from config import get_device
    from extract import load_model_and_tokenizer, load_or_extract_acts
    from intervene import (InterventionSpec, MaskHolder, build_layer_hooks,
                           interventions)
    from tuning import make_schedule, place_bands, trough_layer

    tag = args.model.split("/")[-1]
    device = get_device()
    model, tok = load_model_and_tokenizer(cfg, device)
    acts = load_or_extract_acts(cfg)
    n_layers = int(next(iter(acts.values())).shape[1])
    subs = build_layer_subspaces(acts, range(n_layers), name="diffpca", rank=cfg.rank,
                                 diffpca_angular=True)
    mid_start, mid_end, high_end = place_bands(trough_layer(tag, n_layers), n_layers)
    lam = make_schedule(mid_start, mid_end, args.lam_mid, high_end, args.lam_high, n_layers)
    torch.save({"subspaces": {int(k): v.cpu() for k, v in subs.items()},
                "lam": {int(k): float(v) for k, v in lam.items()},
                "scope": "prompt_only", "n_layers": n_layers}, args.spec_out)
    spec = InterventionSpec(subspaces=subs, lam=lam, scope="prompt_only")
    items, prompts = _prompts(cfg, [l.strip() for l in args.langs.split(",")], args.n, tok)
    print(f"[hf] {len(prompts)} prompts | {len(lam)} edited layers", flush=True)

    out = {}
    for arm, sp in (("clean", None), ("intervened", spec)):
        toks, tops = [], []
        for p in prompts:
            enc = tok(p, return_tensors="pt").to(device)
            mh = MaskHolder(scope="prompt_only")
            hooks = build_layer_hooks(sp, mask_holder=mh) if sp is not None else None
            # prompt_only + a single prefill pass: every position is a prompt token, so the
            # generative branch of the hook (seq>1) edits exactly what vLLM's prefill will.
            ctx = interventions(model, hooks) if hooks else _null()
            with torch.no_grad(), ctx:
                logits = model(**enc).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            top = torch.topk(lp, args.topk)
            toks.append(int(top.indices[0]))
            tops.append([[int(i), round(float(v), 5)] for i, v in zip(top.indices, top.values)])
        out[arm] = {"tokens": toks, "top": tops}
        print(f"[hf] {arm}: first tokens captured", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out))
    print(f"[hf] wrote {args.out}", flush=True)
    return 0


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def emit_vllm(args, cfg):
    import torch
    from intervene import InterventionSpec
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm_backend import VllmRunner

    blob = torch.load(args.spec_in, weights_only=False)
    spec = InterventionSpec(subspaces=blob["subspaces"], lam=blob["lam"], scope=blob["scope"])
    tok = AutoTokenizer.from_pretrained(args.model)
    items, prompts = _prompts(cfg, [l.strip() for l in args.langs.split(",")], args.n, tok)
    # 8192, not 4096: an 8-shot MGSM prompt that does not fit is admitted and returned EMPTY
    # rather than rejected (CLAUDE.md Corrections #4), and this script does not call
    # check_context_window, so an under-sized window would read as a validation failure rather
    # than as a configuration error. The cap costs no VRAM -- the KV pool is sized by
    # gpu_memory_utilization -- so err high, as the repo default does.
    runner = VllmRunner(args.model, max_model_len=8192)
    print(f"[vllm] engine up | {runner.n_layers} layers | {len(prompts)} prompts", flush=True)

    sp_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=args.topk)
    out = {}
    for arm, sp in (("clean", None), ("intervened", spec)):
        with runner.interventions(sp):
            res = runner.llm.generate(prompts, sampling_params=sp_params, use_tqdm=False)
        toks, tops = [], []
        for r in res:
            o = r.outputs[0]
            toks.append(int(o.token_ids[0]))
            lpd = o.logprobs[0] if o.logprobs else {}
            ranked = sorted(lpd.items(), key=lambda kv: -kv[1].logprob)[:args.topk]
            tops.append([[int(k), round(float(v.logprob), 5)] for k, v in ranked])
        out[arm] = {"tokens": toks, "top": tops}
        print(f"[vllm] {arm}: first tokens captured", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out))
    print(f"[vllm] wrote {args.out}", flush=True)
    return 0


def compare(args):
    hf = json.loads(pathlib.Path(args.compare[0]).read_text())
    vl = json.loads(pathlib.Path(args.compare[1]).read_text())
    stats = {}
    for arm in ("clean", "intervened"):
        h, v = hf[arm], vl[arm]
        n = min(len(h["tokens"]), len(v["tokens"]))
        exact = sum(1 for i in range(n) if h["tokens"][i] == v["tokens"][i]) / n
        # top-1 of HF present anywhere in vLLM's top-k (tolerates near-tie reorderings)
        ink = sum(1 for i in range(n)
                  if h["tokens"][i] in {t[0] for t in v["top"][i]}) / n
        stats[arm] = {"n": n, "argmax_match": round(exact, 4), "hf_top1_in_vllm_topk": round(ink, 4)}
        print(f"{arm:11s} n={n}  argmax match={exact:.4f}  hf-top1-in-vllm-topk={ink:.4f}", flush=True)

    floor = stats["clean"]["argmax_match"]
    got = stats["intervened"]["argmax_match"]
    # the intervened arm must not disagree materially MORE than the clean floor does
    drop = floor - got
    ok = drop <= args.max_drop
    print(f"\nclean floor {floor:.4f} -> intervened {got:.4f}  (drop {drop:+.4f}; "
          f"allowed <= {args.max_drop})", flush=True)

    # and the intervention must actually change the first token (else we compared clean twice)
    hchg = sum(1 for a, b in zip(hf["clean"]["tokens"], hf["intervened"]["tokens"]) if a != b)
    vchg = sum(1 for a, b in zip(vl["clean"]["tokens"], vl["intervened"]["tokens"]) if a != b)
    n = len(hf["clean"]["tokens"])
    print(f"[sanity] first token changed by intervention: HF {hchg}/{n} | vLLM {vchg}/{n}", flush=True)
    if vchg == 0:
        print("[FAIL] vLLM intervention was a no-op on the first token")
        ok = False

    pathlib.Path(_root / "results" / "vllm_firsttoken_report.json").write_text(
        json.dumps({"stats": stats, "drop": drop, "hf_changed": hchg,
                    "vllm_changed": vchg, "pass": ok}, indent=2))
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=("hf", "vllm"))
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--langs", default="en,sw")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--lam-mid", type=float, default=0.4)
    ap.add_argument("--lam-high", type=float, default=-0.2)
    ap.add_argument("--max-drop", type=float, default=0.10)
    ap.add_argument("--n-samples", type=int, default=int(os.environ.get("E5_N_SAMPLES", "7500")))
    ap.add_argument("--spec-out", default="results/vllm_ft_spec.pt")
    ap.add_argument("--spec-in", default="results/vllm_ft_spec.pt")
    ap.add_argument("--out", default="results/vllm_ft_arm.json")
    args = ap.parse_args()
    if args.compare:
        return compare(args)
    from config import Config, set_seeds
    set_seeds(0)
    cfg = Config(model_name=args.model, n_samples=args.n_samples)
    return emit_hf(args, cfg) if args.emit == "hf" else emit_vllm(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
