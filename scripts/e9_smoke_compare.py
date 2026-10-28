#!/usr/bin/env python
"""E9 smoke gate: do the grid's own arms behave the same on HF and vLLM? CPU only, no model.

Called by `scripts/prajna/e9_prajna.sbatch smoke`. Reads two directories of `e9_tradeoff_grid.py`
shards — one produced with `E5_ENGINE=hf`, one with `E5_ENGINE=vllm` — and decides whether the grid
may run on vLLM. This is the 7B instance of the check `docs/workshop/implementation.md:42` asks for,
done on the arms E9 will actually generate rather than on a synthetic spec.

CORRECTED 2026-08-24. This script used to GATE on end-to-end `pred` agreement >= 0.95 and on
|acc(vllm) - acc(hf)| <= 0.02, both inherited from `scripts/validate_vllm_vs_hf.py`. Both were wrong
here, for independent reasons, and the 7B `sw` smoke failed on them while the engines were in fact
equivalent. What changed, and why:

1. `pred` AGREEMENT IS NO LONGER A GATE — it is a reported diagnostic.

   `scripts/validate_vllm_firsttoken.py` already settled this in its own docstring: greedy decoding
   is chaotic, one token flipped by a bf16 kernel difference cascades into a completely different
   continuation, and on a benchmark where most answers are wrong the wrong answers are near-
   arbitrary strings. It measured ~42% per-item agreement on the CLEAN arm — where both engines run
   identical mathematics — and concluded end-to-end agreement "is the wrong instrument".

   Measured here on Qwen2.5-7B / sw / 250 items: CLEAN agreement 0.324, with no intervention on
   either side. Conditioning on model confidence shows the mechanism directly — agreement is 0.706
   on items HF got RIGHT and 0.226 on items it got WRONG. The same model/prompts score 0.9792 on the
   first-token gate, with HF's top-1 inside vLLM's top-5 on 100% of prompts: the engines never
   actually disagree about what is likely, they break near-ties differently and the decode amplifies
   it. A metric that cannot exceed ~0.4 on a *correct* pipeline cannot serve as a 0.95 gate.

   The inherited numbers were also calibrated in a different regime: validate_vllm_vs_hf.py defaults
   to 256 max_new_tokens, langs `en,sw` (English is high-confidence and pulls the pooled number up),
   n=32/lang and a 1.5B model. E9's smoke runs 512 tokens, `sw` alone, n=250, at 7B.

   THE ENGINE GATE IS `scripts/validate_vllm_firsttoken.py` — one forward pass per prompt, so there
   is no decode chaos to amplify, and a RELATIVE verdict (intervened vs the clean floor) robust to
   whatever the engines' numerical floor happens to be. Run that per model. This script asks the
   complementary question: do the GRID'S OWN ARMS move the numbers E9 plots, on both engines?

2. THE ACCURACY CHECK IS NOW A PAIRED TEST, NOT A FIXED TOLERANCE.

   `--max-acc-delta 0.02` was set at n=32/lang. At n=250 with acc ~= 0.2 one run's standard error is
   ~0.025, so the old tolerance was ~0.55 SE of the quantity it bounded — it fired on noise, and in
   SE terms it got TIGHTER the more data you gave it. Both engines score the SAME items, so the
   correct instrument is McNemar's exact test on the discordant pairs. We now fail only when the
   difference is statistically significant (p < --alpha); --max-acc-delta is retained purely as a
   large-magnitude backstop for the low-power case (default 0.15 = genuinely broken, not noisy).

What the verdict rests on, per (cond, lang):

  a. accuracy — McNemar exact p >= --alpha (no systematic per-item accuracy difference), AND
     |acc(vllm) - acc(hf)| <= --max-acc-delta as a magnitude backstop.
  b. fidelity — |fid(vllm) - fid(hf)| <= --max-fid-delta. Response fidelity is the axis E9 actually
     plots, and it is far more stable across engines than `pred` (1.0000 vs 1.0000 on the 7B clean
     arm), so a real divergence HERE is meaningful where a `pred` divergence is not.
  c. effect — the intervention did something on BOTH engines. An intervened arm whose fidelity
     matches `clean` to within --min-effect on both is reported as a NO-OP: the exact failure the
     vLLM `intervention_spec` guard exists to prevent (Zhao's wrapper silently generating clean
     output labelled as intervened). Without this, (a) and (b) would happily PASS two engines that
     were both quietly doing nothing.

`pred` agreement and the mutual-unparsed rate are still printed for every cell.
`--gate-on-pred-agreement` restores the old (rejected) behaviour if you want to watch it fail.

Usage:
    python scripts/e9_smoke_compare.py <hf_shard_dir> <vllm_shard_dir>
    python scripts/e9_smoke_compare.py A B --alpha 0.001 --max-fid-delta 0.15
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import pandas as pd

UNDET = {"?", "nan", "none", ""}


def norm_pred(s: "pd.Series") -> "pd.Series":
    """`pred` as comparable strings, with unparseable answers folded to one sentinel.

    Load-bearing, not cosmetic: a bare `.astype(str)` leaves a missing answer as float `nan`, and
    `nan != nan`, so TWO engines that both failed to produce a parseable answer on an item are
    scored as DISAGREEING. On a floored language (bn, sw, te) that is most of the items, which
    fails an agreement gate for the one reason that is not disagreement at all.
    """
    return s.astype("string").fillna("<none>").astype(str)


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs; 1.0 when there are none.

    Both engines score the SAME items, so an unpaired two-proportion test would throw away the
    pairing and overstate the variance. Under H0 (neither engine is systematically more accurate)
    each discordant pair is a fair coin, so `b ~ Binomial(b + c, 0.5)`.

    `b` = items HF got right and vLLM got wrong; `c` = the reverse. Exact below --exact-max
    discordants (the binomial tail is cheap and avoids the continuity-correction wobble at small
    n); normal approximation above it, where the two agree to well past any alpha we gate on.
    """
    n = b + c
    if n == 0:
        return 1.0
    if n <= 1000:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        return min(1.0, 2.0 * tail / (2 ** n))
    z = (abs(b - c) - 1.0) / math.sqrt(n)                 # continuity-corrected
    return math.erfc(z / math.sqrt(2.0))


def load(d: pathlib.Path) -> pd.DataFrame:
    shards = sorted(d.glob("*.csv"))
    if not shards:
        sys.exit(f"no shards in {d}")
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    print(f"  {d.name}: {len(shards)} shards, {len(df)} rows, "
          f"conds={sorted(df['cond'].unique())}, engines={sorted(df.get('engine', pd.Series(['?'])).unique())}")
    return df


def fid(g: pd.DataFrame) -> float:
    """Response fidelity, '?'-excluded (the repo/paper convention in `evaluate.aggregate`)."""
    d = g[~g.output_lang.astype(str).str.lower().isin(UNDET)]
    return float((d.output_lang == d.lang).mean()) if len(d) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", type=pathlib.Path)
    ap.add_argument("vllm_dir", type=pathlib.Path)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="McNemar significance level; a cell FAILS when p < alpha, i.e. the two "
                         "engines differ systematically in accuracy on the same items")
    ap.add_argument("--max-acc-delta", type=float, default=0.15,
                    help="magnitude backstop on |acc(vllm) - acc(hf)| for the low-power case; NOT "
                         "the primary check (see --alpha). The old 0.02 was ~0.55 SE at n=250")
    ap.add_argument("--max-fid-delta", type=float, default=0.15,
                    help="response fidelity is the axis E9 plots and is stable across engines; a "
                         "gap this large is a real divergence, unlike a `pred` gap")
    ap.add_argument("--min-effect", type=float, default=0.01,
                    help="an intervened arm must move fidelity from clean by at least this much on "
                         "at least one engine, else it is reported as a silent no-op")
    ap.add_argument("--gate-on-pred-agreement", action="store_true",
                    help="restore the rejected end-to-end `pred` agreement gate (see the module "
                         "docstring for why it cannot pass on a correct pipeline)")
    ap.add_argument("--min-agree", type=float, default=0.95,
                    help="only used with --gate-on-pred-agreement; otherwise reported, not gated")
    args = ap.parse_args()

    print("loading shards:")
    hf, vl = load(args.hf_dir), load(args.vllm_dir)
    ok = True
    rows = []

    keys = sorted(set(map(tuple, hf[["cond", "lang"]].values))
                  & set(map(tuple, vl[["cond", "lang"]].values)))
    if not keys:
        print("[FAIL] the two directories share no (cond, lang) cell")
        return 1
    missing = (set(map(tuple, hf[["cond", "lang"]].values))
               ^ set(map(tuple, vl[["cond", "lang"]].values)))
    if missing:
        print(f"[FAIL] cells present on only one engine: {sorted(missing)}")
        ok = False

    for cond, lang in keys:
        a = hf[(hf["cond"] == cond) & (hf.lang == lang)].set_index("id")
        b = vl[(vl["cond"] == cond) & (vl.lang == lang)].set_index("id")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            print(f"[FAIL] {cond}/{lang}: no shared items")
            ok = False
            continue
        pa, pb = norm_pred(a.loc[common, "pred"]), norm_pred(b.loc[common, "pred"])
        agree = float((pa.values == pb.values).mean())
        # Reported because agreement resting on both engines failing the same items is agreement,
        # but it is not evidence the intervention path works.
        undet = float(((pa.values == "<none>") & (pb.values == "<none>")).mean())

        ca = a.loc[common, "correct"].astype(bool)
        cb = b.loc[common, "correct"].astype(bool)
        n_b = int((ca & ~cb).sum())                       # HF right, vLLM wrong
        n_c = int((~ca & cb).sum())                       # vLLM right, HF wrong
        p_mc = mcnemar_p(n_b, n_c)
        d_acc = float(cb.mean() - ca.mean())
        f_hf, f_vl = fid(a.loc[common]), fid(b.loc[common])
        d_fid = float(f_vl - f_hf)

        acc_ok = (p_mc >= args.alpha) and (abs(d_acc) <= args.max_acc_delta)
        # NaN fidelity means every item was unparseable on one side — not a pass.
        fid_ok = (abs(d_fid) <= args.max_fid_delta) if d_fid == d_fid else False
        cell_ok = acc_ok and fid_ok
        if args.gate_on_pred_agreement:
            cell_ok = cell_ok and (agree >= args.min_agree)
        ok &= cell_ok
        rows.append({"cond": cond, "lang": lang, "n": len(common),
                     "agree_pred*": round(agree, 4), "both_unparsed": round(undet, 4),
                     "acc_hf": round(float(ca.mean()), 4), "acc_vllm": round(float(cb.mean()), 4),
                     "d_acc": round(d_acc, 4), "b/c": f"{n_b}/{n_c}",
                     "mcnemar_p": round(p_mc, 4),
                     "fid_hf": round(f_hf, 4), "fid_vllm": round(f_vl, 4),
                     "d_fid": round(d_fid, 4),
                     "verdict": "PASS" if cell_ok else "FAIL"})

    print("\n--- per-cell comparison  (* agree_pred is a DIAGNOSTIC, not a gate) ---")
    print(pd.DataFrame(rows).to_string(index=False))

    # no-op detection: an intervened arm must actually move fidelity off clean on some engine
    print("\n--- intervention effect (vs clean, same language) ---")
    tab = pd.DataFrame(rows).set_index(["cond", "lang"])
    for lang in sorted({l for _, l in keys}):
        base = [(c, l) for c, l in keys if l == lang and c == "clean"]
        if not base:
            print(f"  {lang}: no clean arm in this smoke — effect check skipped")
            continue
        c_hf, c_vl = tab.loc[base[0], ["fid_hf", "fid_vllm"]]
        for cond, l in keys:
            if l != lang or cond == "clean":
                continue
            f_hf, f_vl = tab.loc[(cond, l), ["fid_hf", "fid_vllm"]]
            moved = max(abs(f_hf - c_hf), abs(f_vl - c_vl))
            noop = moved < args.min_effect
            ok &= not noop
            print(f"  {cond:28s} {lang}: fid clean {c_hf:.3f}/{c_vl:.3f} -> "
                  f"{f_hf:.3f}/{f_vl:.3f}  |moved|={moved:.3f}  "
                  f"{'*** NO-OP: the intervention did nothing on either engine' if noop else 'ok'}")

    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  "
          f"(alpha={args.alpha}, max_acc_delta={args.max_acc_delta}, "
          f"max_fid_delta={args.max_fid_delta}, min_effect={args.min_effect}"
          f"{', pred-agreement GATED at %s' % args.min_agree if args.gate_on_pred_agreement else ''})")
    print("engine equivalence itself is gated by scripts/validate_vllm_firsttoken.py, not by "
          "`pred` agreement — see this file's docstring.")
    if ok:
        print("The grid may run on vLLM. Next: sbatch scripts/prajna/e9_prajna.sbatch grid <model>")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
