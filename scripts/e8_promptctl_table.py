"""Render the E8 C3 prompt control as one table: languages x (arm x metric).

The arms answer a single question -- can you get multilingual reasoning gains by
*asking*, without touching the residual stream -- so they are only readable side by
side. Baseline first, then the four CoT arms as a 2x2 (English or Chinese exemplars,
answer cue moved or held native), then the two translate-test ceilings.

Reads a committed export bundle, so nothing here needs the cluster or a GPU. Writes a
MultiIndex CSV, a heat-mapped HTML, and the markdown table that docs/ and the paper
quote -- the markdown is generated so its digits cannot drift from the CSV the way the
295383 numbers did.

Every run-specific number in the prose (decode cap, undetectable count, which languages
truncate, the cue contrast) is COMPUTED from the bundle, not typed. Adding a model means
adding a RUNS entry, not editing prose -- the hand-written 1024-cap sentences that used
to live here were true only of Qwen2.5-7B and would have been silently wrong for the
16k-cap DeepSeek run.

  uv run python scripts/e8_promptctl_table.py --tag Qwen2.5-7B-Instruct
  uv run python scripts/e8_promptctl_table.py --tag DeepSeek-R1-Distill-Qwen-7B
  -> results/E8/<tag>/E8_promptctl_table_<tag>.csv   (reload: header=[0,1], index_col=0)
     results/E8/<tag>/E8_promptctl_table_<tag>.html  (open in any browser)
     docs/experiments/E8_promptctl_table_<tag>.md
"""
import argparse
import sys, pathlib, textwrap

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
import pandas as pd
from config import results_path

# One entry per banked c3 run. `decode` is prose because it is the one thing the
# per-item CSV cannot tell you: max_new_tokens is in the data, greedy-vs-sampling is not.
RUNS = {
    "Qwen2.5-7B-Instruct": dict(
        job="297712", date="2026-08-29", bundle="e8c3_promptctl_Qwen2.5-7B-Instruct_2026-08-29",
        decode="greedy", reasoning=False,
    ),
    "DeepSeek-R1-Distill-Qwen-7B": dict(
        job="301051", date="2026-09-02",
        bundle="e8c3_promptctl_DeepSeek-R1-Distill-Qwen-7B_2026-09-02",
        decode="sampling per model card (temperature 0.6, top_p 0.95)", reasoning=True,
    ),
}

# (cond, short label, one-line gloss) -- order is the argument, not alphabetical.
ARMS = [
    ("clean",            "clean",       "native prompt, native cue — the baseline"),
    ("en_cot",           "en cot",      "English exemplars, cue moved to English"),
    ("en_cot_nativecue", "en cot +nc",  "English exemplars, cue held native"),
    ("zh_cot",           "zh cot",      "Chinese exemplars, cue moved to Chinese"),
    ("zh_cot_nativecue", "zh cot +nc",  "Chinese exemplars, cue held native"),
    ("en_question",      "en question", "ceiling: the question itself in English"),
    ("zh_question",      "zh question", "ceiling: the question itself in Chinese"),
]
LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
LOW_RESOURCE = {"bn", "sw", "te", "th"}
METRICS = ["accuracy", "fidelity", "joint"]


def cells(items: pd.DataFrame) -> pd.DataFrame:
    """One row per (lang, arm) with the three headline metrics plus truncation.

    Fidelity is over *detectable* generations, matching `response_fidelity` in the
    agg CSV and scripts/e7_fixed_table.py; joint is strict, so an undetectable
    generation counts against it. The two conventions differ by at most the
    undetectable rate, which `facts()` reports so the doc can say which it used.
    """
    ol = items["output_lang"].astype(str)
    items = items.assign(in_lang=ol == items["lang"], detectable=ol != "?")
    items["joint"] = items["correct"].astype(bool) & items["in_lang"]
    recs = []
    for (lang, cond), g in items.groupby(["lang", "cond"]):
        det = g[g["detectable"]]
        recs.append(dict(
            lang=lang, cond=cond, n=len(g),
            accuracy=g["correct"].mean(),
            fidelity=(det["output_lang"] == lang).mean() if len(det) else float("nan"),
            joint=g["joint"].mean(),
            truncation=g["truncated"].mean(),
        ))
    return pd.DataFrame(recs)


def facts(items: pd.DataFrame, r: pd.DataFrame) -> dict:
    """The run-specific numbers the prose quotes, measured rather than typed.

    `trunc_flag` is the honest version of the old hand-written "te is a truncation
    artifact" note: any (lang, arm) whose truncation differs from that language's
    clean arm by more than 5 points is not being scored on comparable outputs, and
    the reader is told which cells those are instead of trusting a stale sentence.
    """
    caps = sorted(items["max_new_tokens"].unique())
    ol = items["output_lang"].astype(str)
    clean = r[r.cond == "clean"].set_index("lang").truncation
    flags = []
    for _, row in r[r.cond != "clean"].iterrows():
        base = clean.get(row.lang, float("nan"))
        if abs(row.truncation - base) > 0.05:
            flags.append((row.lang, row.cond, base, row.truncation))
    return dict(
        n_items=len(items),
        n_undetectable=int((ol == "?").sum()),
        cap=caps[0] if len(caps) == 1 else None,
        caps=caps,
        max_trunc=float(r.truncation.max()),
        trunc_flag=sorted(flags),
    )


def cue_contrast(r: pd.DataFrame) -> dict:
    """`{p}_cot` minus `{p}_cot_nativecue` -- the cue isolated as one factor.

    Both arms share exemplars and languages, so the difference is the answer cue and
    nothing else. Computed per pivot because the two pivots need not agree.
    """
    out = {}
    for p in ("en", "zh"):
        a, b = r[r.cond == f"{p}_cot"], r[r.cond == f"{p}_cot_nativecue"]
        if len(a) and len(b):
            out[p] = dict(acc=a.accuracy.mean() - b.accuracy.mean(),
                          fid=a.fidelity.mean() - b.fidelity.mean())
    return out


def wide(r: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    t = r.pivot(index="lang", columns="cond", values=metrics).swaplevel(axis=1)
    t = t.reindex(index=LANGS,
                  columns=pd.MultiIndex.from_product([[c for c, _, _ in ARMS], metrics]))
    t.columns.names, t.index.name = ["arm", "metric"], "lang"
    return t


def macro(r: pd.DataFrame) -> pd.DataFrame:
    """Arm means over each arm's own languages, and over clean restricted to match.

    An arm has no cell for its own pivot language, so `clean` is not one number here:
    quoting a mean over 11 against a mean over 10 is the mistake that put a stale
    baseline into three documents. The `clean` column below is always the matched one.
    """
    rows = []
    for cond, label, _ in ARMS:
        sub = r[r.cond == cond]
        langs = sorted(sub.lang)
        base = r[(r.cond == "clean") & (r.lang.isin(langs))]
        rows.append(dict(
            arm=cond, k=len(langs),
            clean_acc=base.accuracy.mean(), acc=sub.accuracy.mean(),
            d_acc=sub.accuracy.mean() - base.accuracy.mean(),
            clean_fid=base.fidelity.mean(), fid=sub.fidelity.mean(),
            clean_joint=base.joint.mean(), joint=sub.joint.mean(),
            d_joint=sub.joint.mean() - base.joint.mean(),
        ))
    return pd.DataFrame(rows).set_index("arm")


def heat(v, lo_is_bad=True):
    if v != v:
        return "#eee"
    return f"hsl({v * 120:.0f},70%,87%)" if lo_is_bad else "#fff"


def html(t: pd.DataFrame, tag: str, run: dict, f: dict) -> str:
    head = "".join(f'<th colspan="{len(METRICS)}">{lab}</th>' for _, lab, _ in ARMS)
    sub = "".join(f"<th>{m[:3]}</th>" for _ in ARMS for m in METRICS)
    body = []
    for lang in t.index:
        tds = "".join(
            f'<td style="background:{heat(t.loc[lang, (c, m)])}">'
            f'{"·" if t.loc[lang, (c, m)] != t.loc[lang, (c, m)] else f"{t.loc[lang, (c, m)]:.3f}"}</td>'
            for c, _, _ in ARMS for m in METRICS)
        mark = " *" if lang in LOW_RESOURCE else ""
        body.append(f"<tr><th>{lang}{mark}</th>{tds}</tr>")
    cap = f"{f['cap']}-token cap" if f["cap"] else f"caps {f['caps']}"
    return (f"<!doctype html><meta charset=utf-8><title>E8 prompt control — {tag}</title>"
            "<style>body{font:13px/1.5 system-ui;padding:24px}table{border-collapse:collapse}"
            "th,td{border:1px solid #ccc;padding:3px 7px;text-align:right}"
            "th{background:#f4f4f4}caption{text-align:left;padding-bottom:8px;color:#444}</style>"
            f"<table><caption>E8 C3 prompt control — {tag}, job {run['job']}, n=250/cell, {cap}."
            " Green is high. `·` = no cell (an arm has no row for its own pivot language)."
            " * = low-resource.</caption>"
            f"<tr><th></th>{head}</tr><tr><th>lang</th>{sub}</tr>{''.join(body)}</table>")


def md_table(t: pd.DataFrame, metric: str) -> list[str]:
    out = ["| lang | " + " | ".join(lab for _, lab, _ in ARMS) + " |",
           "|---" * (len(ARMS) + 1) + "|"]
    for lang in t.index:
        vals = []
        for c, _, _ in ARMS:
            v = t.loc[lang, (c, metric)]
            vals.append("·" if v != v else f"{v:.3f}")
        out.append(f"| `{lang}`{' *' if lang in LOW_RESOURCE else ''} | " + " | ".join(vals) + " |")
    return out


def para(text: str) -> list[str]:
    """Wrap a generated paragraph to the width the hand-written docs here use."""
    return textwrap.fill(" ".join(text.split()), width=95).splitlines()


def markdown(t: pd.DataFrame, m: pd.DataFrame, sig: dict, tag: str, run: dict,
             f: dict, cue: dict) -> str:
    cap = f"{f['cap']}-token decode cap" if f["cap"] else f"decode caps {f['caps']}"
    undet_pct = f["n_undetectable"] / f["n_items"]
    L = [f"# E8 · C3 prompt control — {tag}, per language",
         "",
         "**Generated by `scripts/e8_promptctl_table.py`. Do not hand-edit** — regenerate it, so",
         "these digits cannot drift from the CSV the way the superseded 295383 numbers did.",
         "",
         f"Job **{run['job']}** ({run['date']}), MGSM all-250 split, n=250 per cell, {cap},",
         f"{run['decode']}, vLLM. Source: `exports/{run['bundle']}/`.",
         "",
         "The question these arms answer is whether multilingual reasoning gains can be had by",
         "*asking* — no residual-stream edit — and they are only readable side by side. `clean` is",
         "the baseline; the four CoT arms are a 2×2 of exemplar language against whether the answer",
         "cue moves with the exemplars or is held native; the two `question` arms are translate-test",
         "ceilings, where the problem itself is presented in the pivot language.",
         "",
         "**`·` means the cell does not exist**: an arm has no row for its own pivot language, so",
         "there is no `en` row under an `en_*` arm and no `zh` row under a `zh_*` arm. This is why",
         "no single `clean` mean is comparable to every arm — see the summary table.",
         ""]
    for cond, lab, gloss in ARMS:
        L.append(f"- **{lab}** (`{cond}`) — {gloss}")
    L += ["", "`*` marks the four low-resource languages.", "",
          "## Accuracy", "", *md_table(t, "accuracy"), "",
          "## Response fidelity — the fraction answering *in* the asked language", "",
          *para("Over detectable generations only, matching `response_fidelity` in the agg CSV. "
                f"{f['n_undetectable']} of {f['n_items']:,} generations are undetectable "
                f"({undet_pct:.1%}), so that is the most the convention can move a cell — it is "
                "not the same convention as `joint`, which counts an undetectable generation "
                "against you."),
          "", *md_table(t, "fidelity"), "",
          "## Joint — correct **and** answered in the asked language", "",
          "The metric the paper leads with, included because accuracy and fidelity are not",
          "independent here: every arm that buys accuracy pays for it in fidelity, and joint is",
          "where that shows up as one number.",
          "", *md_table(t, "joint"), "",
          "## Arm means, each against the `clean` cells that match it", ""]
    L += ["| arm | k | clean acc | acc | Δacc | clean fid | fid | joint | Δjoint |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cond, lab, _ in ARMS:
        r = m.loc[cond]
        d = f"{r.d_acc:+.3f}" + (f" {sig[cond]}" if cond in sig else "")
        L.append(f"| {lab} | {int(r.k)} | {r.clean_acc:.3f} | {r.acc:.3f} | {d} | "
                 f"{r.clean_fid:.3f} | {r.fid:.3f} | {r.joint:.3f} | {r.d_joint:+.3f} |")
    L += ["",
          "The `Δacc` column carries, in brackets, how many of that arm's languages have a paired",
          "bootstrap CI excluding zero (up/down) from `promptctl_paired_*.csv`.",
          "",
          "## Two things to carry when quoting this table",
          ""]

    if f["trunc_flag"]:
        L += para(
            "**Truncation is not comparable in every cell.** These (lang, arm) cells truncate "
            f"more than 5 points away from that language's own `clean` arm at this {cap}, so the "
            "arms are not being scored on comparable outputs there and the cell should be "
            "excluded from any claim about which languages the prompt helps:") + [""]
        for lang, cond, base, tr in f["trunc_flag"]:
            L.append(f"- `{lang}` / `{cond}` — {tr:.1%} against clean {base:.1%}")
        L.append("")
    else:
        L += para(
            "**No cell is truncation-confounded.** The worst truncation rate anywhere in the "
            f"table is {f['max_trunc']:.1%}, and no arm differs from its language's `clean` arm by "
            f"more than 5 points, so at this {cap} every arm is scored on outputs that ran to "
            "completion. This is the check that has to be re-run per model, not inherited: the "
            "same experiment on Qwen2.5-7B at a 1024-token cap truncated 21.6% of `te`'s clean "
            "arm, and its `te` CoT gains were an artifact of that.") + [""]

    if cue:
        parts = ", ".join(f"`{p}` {v['acc']:+.3f} accuracy for {v['fid']:+.3f} fidelity"
                          for p, v in cue.items())
        L += para(
            "**The cue is the one-factor experiment.** Reading the 2×2 down its columns is the "
            "point of running it: `cot` and `cot_nativecue` share exemplars and languages, so "
            "their difference isolates the answer cue and nothing else. Here that contrast is "
            f"{parts}. Whatever accuracy the prompt buys is bought by substituting the language "
            "— which is the same mechanism the E9 intervention grid shows, reached here without "
            "any representation claim.") + [""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="Qwen2.5-7B-Instruct", choices=sorted(RUNS),
                    help="model tag; must have a RUNS entry naming its export bundle")
    args = ap.parse_args()
    tag, run = args.tag, RUNS[args.tag]

    bundle = _root / "exports" / run["bundle"] / "data"
    items = pd.read_csv(bundle / f"promptctl_items_{tag}.csv.gz")
    r = cells(items)
    f = facts(items, r)
    t = wide(r, METRICS + ["truncation"])

    out_csv = results_path(f"E8_promptctl_table_{tag}.csv", tag)
    t.round(4).to_csv(out_csv)
    out_html = results_path(f"E8_promptctl_table_{tag}.html", tag)
    out_html.write_text(html(t, tag, run, f))

    paired = pd.read_csv(bundle / f"promptctl_paired_{tag}.csv")
    sig = {}
    for cond, _, _ in ARMS:
        g = paired[paired.contrast == f"{cond}-clean"]
        if len(g):
            sig[cond] = f"[{(g.delta_lo > 0).sum()}↑ {(g.delta_hi < 0).sum()}↓ of {len(g)}]"
    doc = _root / "docs" / "experiments" / f"E8_promptctl_table_{tag}.md"
    doc.write_text(markdown(t, macro(r), sig, tag, run, f, cue_contrast(r)))

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(f"wrote {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
