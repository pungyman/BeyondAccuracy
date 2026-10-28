#!/usr/bin/env python
"""C3 -- the prompt-level control: what does a one-line prompt change buy, with no intervention?

Task C3 of docs/workshop/proposal_and_plan.md. The paper argues that language-subspace ablation
raises MGSM accuracy by moving the model toward answering in English. The obvious question a
reviewer asks next is: how much of that could you get by simply *asking*? If a prompt change
matches the intervention's accuracy gain, the intervention's accuracy story is fully deflated,
because a prompt costs nothing and does not pretend to be a claim about representations.

Three conditions, NO intervention anywhere -- no hooks, no subspace, no activation cache. This
script therefore needs only the model, and can run before/independently of any extraction.

Conditions are generated from a list of PIVOT languages (`PC_PIVOTS`, default `en`). For each
pivot `p`, on every language except `p` itself:

  clean               the paper's own baseline: native `n_shot` localized exemplars, localized
                      answer cue. Re-run here rather than reused so all arms share one decode path.
  {p}_cot             native-language QUESTION, but PIVOT exemplars and the pivot answer cue --
                      "reason and answer in English about this Bengali problem". The cheap
                      intervention-free way to move the reasoning language. (Standard MGSM en-CoT
                      when p=en.)
  {p}_cot_nativecue   pivot exemplars, NATIVE cue (`PC_NATIVE_CUE=1`). `{p}_cot` moves the
                      exemplars AND the cue, and the cue is the last token before generation --
                      the strongest single signal of output language. This arm moves only the
                      exemplars, so the two effects can be told apart.
  {p}_question        the PIVOT-LANGUAGE TRANSLATION of the same problem, with pivot exemplars --
                      the translate-everything upper bound. Free because MGSM is parallel, so
                      problem i in every language is the same problem.

The comparison of interest is `{p}_cot - clean` in ACCURACY against the intervention's accuracy
gains, and `{p}_cot`'s response fidelity -- the same substitution, honestly labelled as a prompt
choice rather than as disentanglement.

**Why a pivot list and not just English (2026-08-25).** The arm set was specced when the
substitution target was believed to be English. It is not: the E9 grid sends output to `zh` at
96-100%, and this script's own 2026-08-24 outputs agree -- 63% of `en_cot`'s off-target responses
were Chinese, not English (bn 85%, te 95%, th 95%), verified as genuine Han script. The zh
attractor is therefore not an artifact of representation surgery; it appears when the language
signal is disturbed by prompting too. `PC_PIVOTS=en,zh` runs both, with `en` still the ceiling arm
because it is the model's strongest language (clean en 0.872 vs zh 0.812).

Scored on the frozen test half by default; `PC_SPLIT=all` scores all 250, which is what E9 uses.
The split protects a TUNED hyperparameter and C3 tunes nothing, so `all` is correct whenever these
numbers meet E9's on one axis -- the frozen half runs ~3 points easier (0.634 vs 0.604).

New campaign options: PC_CHUNK_SIZE=25 enables atomic resumable generation shards;
PC_RUN_LABEL keeps runs separate; PC_SEED selects the sampling replicate;
PC_REQUIRE_NATIVE=1 adds an explicit target-language instruction with pivot prompts;
PC_CONDITIONS selects a comma-separated subset including clean; PC_N caps items per language.
Sharded runs use deterministic per-shard seeds and are a new protocol, not bitwise reproductions
of historical unsharded runs. Every new run also writes strict per-language joint scores.

Output: results/E8/promptctl_items_<tag>.csv   per-item rows, all three conditions
        results/E8/promptctl_agg_<tag>.csv     per (task, lang, cond) acc + fidelity + CI
        results/E8/promptctl_paired_<tag>.csv  en_cot - clean and en_question - clean, paired
                                               (which is which: the `contrast` column)

Knobs: E5_MODEL, PC_LANGS, PC_NSHOT (default 8, matching the rest of the paper), HTH_DTYPE,
       E5_EVAL_BATCH, E5_EVAL_MAX_BATCH_TOKENS, PC_MAX_NEW (see below), E5_ENGINE=hf|vllm
       (default hf), PC_SPLIT=test|all, PC_PIVOTS (default `en`), PC_NATIVE_CUE=1, PC_DRYRUN=1.
       PC_DRYRUN exercises the whole model-free construction and its asserts -- use it before
       spending a GPU.

ENGINE. `E5_ENGINE=vllm` routes the three conditions through the production seam in
`evaluate.run_eval` -- legal here because all three arms are UNHOOKED (`hooks=None`), so the
fail-loud hooks-without-spec guard never trips and no intervention can vanish silently. No HF model
is loaded on that path (its VRAM goes to the engine), matching `scripts/e9_tradeoff_grid.py`. The KV
window is sized from the ACTUAL longest prompt across all three conditions and floored at 8192: the
long pole is the native 8-shot `te` prompt (~4.9k tokens on Qwen2.5's byte-fallback BPE), and at
4096 vLLM admits an over-long prompt and returns zero tokens with finish_reason="length" -- silent
empties, not an error. NOTE `HTH_DTYPE` is a no-op on the vLLM path: `evaluate._get_vllm_runner`
passes no dtype, so `VllmRunner` uses its own bfloat16 default (identical to this script's default).
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, replace
import pathlib
import sys

_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
             if (p / "src" / "config.py").exists())
sys.path.insert(0, str(_root / "src"))

import pandas as pd
import torch

from benchmarks import BenchItem, _load_hf_split, eval_languages, split_dev_test
from config import Config, get_device, set_seeds
from evaluate import aggregate, paired_delta, run_eval
from extract import load_model_and_tokenizer
from models import model_spec
from prompts import build_mgsm_prompt
from prompt_checkpoint import ensure_manifest, exclusive_run, fingerprint, shard_seed

SEED = int(os.environ.get("PC_SEED", "0"))
set_seeds(SEED)

_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
    os.environ.get("HTH_DTYPE", "bf16")]
cfg = Config(model_name=os.environ.get("E5_MODEL", "Qwen/Qwen2.5-7B-Instruct"), dtype=_DT)
TAG = cfg.model_name.split("/")[-1]
N_SHOT = int(os.environ.get("PC_NSHOT", "8"))
# Generation cap. An explicit PC_MAX_NEW always wins; otherwise the per-model registry decides.
# This used to be a flat `PC_MAX_NEW` default of 512 passed EXPLICITLY to run_eval, which silently
# overrode the registry -- fatal for a reasoning model. `run_eval` resolves 16384 for Qwen3 and the
# R1-distills, but only when the caller passes nothing; passing 1024 clips the `<think>` trace, and
# `scoring.strip_reasoning(expect_trace=True)` then returns "" rather than scoring a number out of
# mid-reasoning. Every such item lands as WRONG, not as truncated, so the failure is silent and the
# whole run reads as a near-zero-accuracy model.
# For non-reasoning models the registry supplies no cap and 512 stands -- but raise it to 1024 for
# any run including `te` or `bn`, whose clean truncation is 0.78 / 0.37 at 512 (e8_prajna.sbatch
# does this for you).
_SPEC = model_spec(cfg.model_name)
_PC_MAX_NEW_ENV = os.environ.get("PC_MAX_NEW")
MAX_NEW = int(_PC_MAX_NEW_ENV) if _PC_MAX_NEW_ENV else (_SPEC.max_new_tokens or 512)

# Output tag. All three CSVs key on the model tag alone, so the SAME model run twice -- Qwen3
# thinking-ON and thinking-OFF -- would overwrite each other's results, and would interleave into a
# corrupt file if the two jobs ran concurrently. When a mode is asked for explicitly, stamp it.
# Left bare when E5_ENABLE_THINKING is unset, so every committed artifact keeps its filename:
# Qwen2.5-7B and the forced-`<think>` models (R1-distill, GLM-Z1) never set it.
_THINK_ENV = os.environ.get("E5_ENABLE_THINKING")
OUT_TAG = TAG if _THINK_ENV is None else (
    f"{TAG}_{'think' if _THINK_ENV not in ('', '0', 'false', 'False') else 'nothink'}")
RUN_LABEL = os.environ.get("PC_RUN_LABEL", "").strip()
if RUN_LABEL and not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_LABEL):
    raise ValueError("PC_RUN_LABEL must contain only letters, digits, underscore or hyphen")
if RUN_LABEL:
    OUT_TAG += "_" + RUN_LABEL
elif SEED:
    OUT_TAG += f"_seed{SEED}"
CHUNK = int(os.environ.get("PC_CHUNK_SIZE", "0"))
if CHUNK < 0:
    raise ValueError("PC_CHUNK_SIZE must be nonnegative")
REQUIRE_NATIVE = os.environ.get("PC_REQUIRE_NATIVE", "0") == "1"
LANGUAGE_NAMES = dict(bn="Bengali", de="German", en="English", es="Spanish", fr="French",
                      ja="Japanese", ru="Russian", sw="Swahili", te="Telugu", th="Thai", zh="Chinese")
# Which problems to score. `test` = the frozen seeded half (125/lang), the split every A1 number
# uses; `all` = all 250, which is what E9 scores. The split exists to protect a TUNED
# hyperparameter (lambda was selected on dev in E6) and C3 tunes nothing -- three fixed prompts, no
# search, exemplars drawn from the `train` split -- so `all` contaminates nothing and is the right
# setting whenever these numbers will be compared with E9's. Default stays `test` so the committed
# 2026-08-24 run reproduces byte-for-byte.
SPLIT = os.environ.get("PC_SPLIT", "test").strip().lower()
if SPLIT not in ("test", "all"):
    raise SystemExit(f"PC_SPLIT must be 'test' or 'all', got {SPLIT!r}")
# Pivot languages: the language whose exemplars (and cue) replace the native ones. Historically
# hardcoded to English, on the belief that ablation pushes output toward English. The E9 trace
# refuted that -- the interventions land in `zh` at 96-100% -- and C3's own outputs agree: 63% of
# en_cot's off-target responses are Chinese, not English. So the pivot is now a list.
PIVOTS = [x.strip() for x in os.environ.get("PC_PIVOTS", "en").split(",") if x.strip()]
# `{p}_cot` swaps BOTH the exemplars and the final answer cue. The cue is the last token before
# generation and the strongest single signal of output language, so that arm confounds two things.
# `{p}_cot_nativecue` holds the cue native and moves only the exemplars, separating them.
NATIVE_CUE = os.environ.get("PC_NATIVE_CUE", "") not in ("", "0")
ENGINE = os.environ.get("E5_ENGINE", "hf").strip().lower()
DRYRUN = os.environ.get("PC_DRYRUN", "") not in ("", "0")
# Recompute the agg / paired tables (and the stdout summaries) from an EXISTING per-item CSV,
# skipping generation entirely. CPU-only, no model, no engine. Two uses: recovering a run whose
# generation succeeded but whose post-processing raised -- the per-item CSV is written first, so
# nothing needs regenerating -- and re-running the statistics after changing them. Deliberately
# reuses this script's own tail rather than living in a separate file, so the two cannot drift.
FROM_ITEMS = os.environ.get("PC_FROM_ITEMS", "").strip()
OUT = _root / "results" / "E8"
OUT.mkdir(parents=True, exist_ok=True)

_LANGS_ENV = os.environ.get("PC_LANGS")
LANGS = [l.strip() for l in (_LANGS_ENV or ",".join(eval_languages("mgsm", cfg))).split(",")
         if l.strip()]
if not _LANGS_ENV:                    # en is the default reference, not a condition to translate
    LANGS = [l for l in LANGS if l != "en"]   # into; pass PC_LANGS explicitly to override


def exemplars(lang: str) -> list[dict]:
    """The same first-`n_shot` train records `benchmarks._load_mgsm` uses, so `clean` here is
    byte-identical to the prompts behind the A1 numbers."""
    if N_SHOT <= 0:
        return []
    train = _load_hf_split("mgsm", lang, "train")
    return [{"question": r["question"], "answer": r.get("answer") or ""}
            for r in list(train)[:N_SHOT]]


def build_items() -> dict[str, list[BenchItem]]:
    """One item list per condition, id-aligned to `clean` within that condition's language set.

    `clean` covers every language in LANGS. For each pivot `p`, `{p}_cot` / `{p}_question` (plus
    `{p}_cot_nativecue` when PC_NATIVE_CUE) cover LANGS **minus p itself**: on its own language a
    pivot arm is byte-identical to `clean`, so generating it would be n wasted rows that then plot
    as a spurious dot. Conditions therefore may differ in length; `main` checks each against the
    matching subset of `clean` rather than requiring one global length, and `paired_delta` pairs
    within `(model, task, lang)` so uneven coverage is fine.
    """
    piv_ex = {p: exemplars(p) for p in PIVOTS}
    piv_rows = {p: list(_load_hf_split("mgsm", p, "test")) for p in PIVOTS}
    conds: dict[str, list[BenchItem]] = {"clean": []}
    for p in PIVOTS:
        conds[f"{p}_cot"] = []
        conds[f"{p}_question"] = []
        if NATIVE_CUE:
            conds[f"{p}_cot_nativecue"] = []

    for lang in LANGS:
        native_ex = exemplars(lang)
        rows = list(_load_hf_split("mgsm", lang, "test"))
        for p, pr in piv_rows.items():
            if len(rows) != len(pr):
                raise ValueError(f"{lang}: {len(rows)} test rows vs {p} {len(pr)} -- MGSM should "
                                 f"be parallel; {p}_question alignment would be wrong")
        built: dict[str, list[BenchItem]] = {k: [] for k in conds}
        for i, r in enumerate(rows):
            gold, meta = str(r["answer_number"]), {"id": f"mgsm-{lang}-{i}", "n_shot": N_SHOT}
            # clean: exactly what benchmarks._load_mgsm produces
            built["clean"].append(BenchItem(
                task="mgsm", lang=lang, gold=gold, meta={**meta, "instr_lang": lang},
                prompt=build_mgsm_prompt({"question": r["question"], "lang": lang},
                                         native_ex, n_shot=N_SHOT, instr_lang=lang)))
            for p in PIVOTS:
                if lang == p:
                    continue                     # degenerate: identical to `clean`
                # {p}_cot: native question, pivot exemplars AND pivot answer cue
                built[f"{p}_cot"].append(BenchItem(
                    task="mgsm", lang=lang, gold=gold, meta={**meta, "instr_lang": p},
                    prompt=build_mgsm_prompt({"question": r["question"], "lang": p},
                                             piv_ex[p], n_shot=N_SHOT, instr_lang=p)))
                # {p}_cot_nativecue: pivot exemplars, NATIVE cue -- the cue/exemplar separator.
                # `build_mgsm_prompt` picks the cue off item["lang"], so leaving that native while
                # passing pivot exemplars is the whole change.
                if NATIVE_CUE:
                    built[f"{p}_cot_nativecue"].append(BenchItem(
                        task="mgsm", lang=lang, gold=gold,
                        meta={**meta, "instr_lang": lang, "exemplar_lang": p},
                        prompt=build_mgsm_prompt({"question": r["question"], "lang": lang},
                                                 piv_ex[p], n_shot=N_SHOT, instr_lang=lang)))
                # {p}_question: the pivot-language twin of the SAME problem index -- the
                # translate-everything ceiling, free because MGSM is parallel.
                #
                # NB this prompt does not depend on `lang`, so {p}_question is ONE number, not a
                # per-language curve: the same problems decoded once per language, identical up to
                # engine nondeterminism. KEEP the duplication -- the per-language copies are what
                # let evaluate.paired_delta pair within (model, task, lang).
                p_r = piv_rows[p][i]
                if str(p_r["answer_number"]) != gold:
                    raise ValueError(f"{lang}[{i}]: gold {gold} != {p} twin "
                                     f"{p_r['answer_number']} -- the parallel-index assumption is "
                                     f"violated, refusing to proceed")
                built[f"{p}_question"].append(BenchItem(
                    task="mgsm", lang=lang, gold=gold, meta={**meta, "instr_lang": p},
                    prompt=build_mgsm_prompt({"question": p_r["question"], "lang": p},
                                             piv_ex[p], n_shot=N_SHOT, instr_lang=p)))
        for k, v in built.items():
            if not v:
                continue                         # this language contributes nothing to that arm
            # `split_dev_test` keys on meta["id"], whose prefix is constant within a language, so
            # the seeded permutation picks the SAME indices in every language and every condition.
            conds[k].extend(v if SPLIT == "all" else split_dev_test(v, dev_frac=0.5, seed=0)[1])
    if REQUIRE_NATIVE:
        for p in PIVOTS:
            conds[f"{p}_cot_target"] = [replace(it, prompt=(
                f"Give your final answer and its explanation in {LANGUAGE_NAMES[it.lang]}.\n\n"
                + it.prompt)) for it in conds[f"{p}_cot"]]
    selected = os.environ.get("PC_CONDITIONS", "").strip()
    if selected:
        names = [x.strip() for x in selected.split(",")]
        if "clean" not in names or set(names) - set(conds):
            raise ValueError("PC_CONDITIONS must include clean and name existing conditions")
        conds = {k: conds[k] for k in names}
    limit = int(os.environ.get("PC_N", "0"))
    if limit < 0:
        raise ValueError("PC_N must be nonnegative")
    if limit:
        conds = {k: [it for lang in LANGS for it in [x for x in v if x.lang == lang][:limit]]
                 for k, v in conds.items()}
    return conds

def main() -> None:
    if FROM_ITEMS:
        items_df = pd.read_csv(FROM_ITEMS)
        print(f"PC_FROM_ITEMS={FROM_ITEMS}: loaded {len(items_df)} rows, "
              f"{items_df.cond.nunique()} conditions, {items_df.lang.nunique()} languages "
              f"-- skipping generation")
        finalize(items_df)
        return
    conds = build_items()
    n = len(conds["clean"])
    per_lang = n // max(1, len(LANGS))
    print(f"model={cfg.model_name} engine={ENGINE} langs={LANGS} n_shot={N_SHOT} "
          f"out_tag={OUT_TAG} "
          f"max_new={MAX_NEW} ({'PC_MAX_NEW' if _PC_MAX_NEW_ENV else 'registry'}) "
          f"reasoning={_SPEC.is_reasoning} sampling={_SPEC.gen.do_sample} "
          f"split={SPLIT} pivots={PIVOTS} native_cue={NATIVE_CUE}")
    print(f"clean: {n} items ({per_lang} per language x {len(LANGS)} languages)")
    # Conditions may legitimately differ in length -- a pivot arm skips its own language -- so the
    # gate is per-condition against the matching subset of `clean`, not one global length.
    ids = {k: [it.meta["id"] for it in v] for k, v in conds.items()}
    for k, items in conds.items():
        if k == "clean":
            continue
        langs_k = {it.lang for it in items}
        expect = [i for i, it in zip(ids["clean"], conds["clean"]) if it.lang in langs_k]
        assert ids[k] == expect, (f"{k}: id order/coverage diverged from clean "
                                  f"({len(ids[k])} ids over {sorted(langs_k)})")
        print(f"{k}: {len(items)} items over {len(langs_k)} languages")
    total = sum(len(v) for v in conds.values())
    print(f"total: {total} generations across {len(conds)} conditions")
    # Everything above is model-free, and the two asserts ARE the correctness gate on the three-way
    # construction, so PC_DRYRUN exercises them offline -- no weights, no GPU, no engine.
    if DRYRUN:
        print(f"PC_DRYRUN: {total} generations over {len(conds)} conditions "
              f"-- exiting before any model/engine load")
        return

    device = get_device()
    if ENGINE == "vllm":
        # No HF model: run_eval ignores it on this path, so that VRAM goes to the engine. Size the
        # KV window from the ACTUAL longest prompt over ALL THREE conditions and never below 8192
        # -- see the module docstring. `setdefault` so an explicit env still wins.
        from transformers import AutoTokenizer

        from corpus import build_chat_inputs
        model, tok = None, AutoTokenizer.from_pretrained(cfg.model_name)
        longest = 0
        for _items in conds.values():
            enc = tok(build_chat_inputs(tok, [it.prompt for it in _items]))["input_ids"]
            longest = max(longest, max(len(x) for x in enc))
        want_len = max(8192, longest + MAX_NEW + 128)
        os.environ.setdefault("E5_VLLM_MAX_MODEL_LEN", str(want_len))
        print(f"vLLM: longest prompt {longest} tok + {MAX_NEW} new -> "
              f"E5_VLLM_MAX_MODEL_LEN={os.environ['E5_VLLM_MAX_MODEL_LEN']} (floor 8192)",
              flush=True)
    else:
        model, tok = load_model_and_tokenizer(cfg, device)
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ("torch", "transformers", "vllm", "langdetect", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    frames = []
    checkpoint = None
    if CHUNK:
        manifest = dict(versions=versions, model=cfg.model_name,
                        revision=os.environ.get("LRD_MODEL_REVISION") or getattr(tok, "init_kwargs", {}).get("_commit_hash"),
                        engine_generation_config=(model.generation_config.to_dict()
                            if model is not None and hasattr(model, "generation_config") else None),
                        tokenizer=fingerprint(tok.get_vocab()), template=str(tok.chat_template),
                        engine=ENGINE, seed=SEED, chunk_size=CHUNK, max_new=MAX_NEW,
                        thinking=_THINK_ENV, decode=asdict(_SPEC.gen), dtype=str(cfg.dtype),
                        keep_trace=os.environ.get("E5_KEEP_TRACE", ""),
                        batch=os.environ.get("E5_EVAL_BATCH"),
                        batch_tokens=os.environ.get("E5_EVAL_MAX_BATCH_TOKENS"),
                        prompts=fingerprint({k: [(it.meta["id"], it.prompt, it.gold) for it in v]
                                             for k, v in conds.items()}),
                        code=fingerprint({name: (_root / name).read_text() for name in
                            ("scripts/e8_prompt_control.py", "src/extract.py", "src/evaluate.py", "src/vllm_backend.py",
                             "src/scoring.py", "src/corpus.py", "src/models.py", "src/prompt_checkpoint.py")}))
        checkpoint = ensure_manifest(OUT / "promptctl_shards" / OUT_TAG, manifest)
    with exclusive_run(checkpoint):
        for cond, items in conds.items():
            groups = [("all", items)] if not CHUNK else [
                (lang, [it for it in items if it.lang == lang]) for lang in LANGS]
            for lang, group in groups:
                size = CHUNK or max(1, len(group))
                for start in range(0, len(group), size):
                    batch = group[start:start + size]
                    path = checkpoint / f"{cond}_{lang}_{start:05d}.csv.gz" if checkpoint else None
                    if path and path.exists():
                        df = pd.read_csv(path, keep_default_na=False)
                        if df.id.tolist() != [it.meta["id"] for it in batch] or set(df.cond) != {cond}:
                            raise ValueError(f"Invalid checkpoint rows: {path}")
                    else:
                        chunk_seed = shard_seed(SEED, cond, lang, start) if CHUNK else SEED
                        if CHUNK or SEED:
                            set_seeds(chunk_seed)
                        print(f"--- {cond}/{lang}/{start}: {len(batch)} items seed={chunk_seed}", flush=True)
                        df = run_eval(model, tok, batch, hooks=None, cond=cond, device=device,
                                      max_new_tokens=MAX_NEW, model_name=cfg.model_name,
                                      gen=replace(_SPEC.gen, seed=chunk_seed),
                                      batch_size=int(os.environ["E5_EVAL_BATCH"]) if os.environ.get("E5_EVAL_BATCH") else None,
                                      max_batch_tokens=int(os.environ["E5_EVAL_MAX_BATCH_TOKENS"])
                                      if os.environ.get("E5_EVAL_MAX_BATCH_TOKENS") else None)
                        df["run_seed"] = SEED
                        df["generation_seed"] = chunk_seed
                        if path:
                            tmp = path.with_suffix(".tmp")
                            df.to_csv(tmp, index=False, compression="gzip")
                            tmp.replace(path)
                    frames.append(df)
        items_df = pd.concat(frames, ignore_index=True)
        dest = OUT / f"promptctl_items_{OUT_TAG}.csv"
        tmp = dest.with_suffix(".tmp")
        items_df.to_csv(tmp, index=False)
        tmp.replace(dest)
        finalize(items_df)


def finalize(items_df: pd.DataFrame) -> None:
    """Everything downstream of generation: the agg and paired tables, and the stdout summaries.

    Split out so `PC_FROM_ITEMS` can re-enter here against a per-item CSV that already exists. The
    per-item file is written by `main` BEFORE this runs, so a raise in here never costs generations.
    """
    # Strict fidelity and joint accuracy: unidentified output languages count as failures.
    strict = items_df.copy()
    strict["fidelity_strict"] = strict.output_lang.eq(strict.lang)
    if strict.correct.dtype == object:
        strict["correct"] = strict.correct.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if strict.correct.isna().any():
            raise ValueError("Unrecognized correctness labels")
    strict["joint"] = strict.correct.astype(bool) & strict.fidelity_strict
    strict.groupby(["model", "lang", "cond"]).agg(
        n=("correct", "size"), accuracy=("correct", "mean"),
        fidelity_strict=("fidelity_strict", "mean"), joint=("joint", "mean")
    ).reset_index().to_csv(OUT / f"promptctl_joint_{OUT_TAG}.csv", index=False)
    # aggregate() groups by (model, task, lang) ONLY and copies `cond` from the first row, so it
    # must be called per condition -- passing a multi-condition frame silently mislabels rows.
    # Same pattern as e7_fixed_headtohead.py / e6_phase2_behavioral.py.
    agg = pd.concat([aggregate(g).assign(cond=c) for c, g in items_df.groupby("cond")],
                    ignore_index=True)
    agg.to_csv(OUT / f"promptctl_agg_{OUT_TAG}.csv", index=False)
    # `paired_delta` emits no column naming the comparison, so stamp `contrast` ourselves before
    # concatenating: without it the two blocks of rows are indistinguishable in the CSV and the
    # reader has to know the concat order to tell an en_cot delta from an en_question one.
    # `paired_delta` groups by (model, task, lang) over cond_a -- `clean`, which covers EVERY
    # language -- and raises if a group does not pair 1:1 on `id`. A pivot arm skips its own
    # language, so pairing it against the full clean set raises on that language. Subset `clean` to
    # the arm's own languages here rather than weakening the guard: that 1:1 check is what stops a
    # silent mis-pairing from corrupting the headline delta, and it should stay strict.
    # Arms come from the DATA, not from a `conds` dict -- PC_FROM_ITEMS has no such dict, and
    # deriving them here means the tables always describe the rows actually present. `unique()`
    # preserves order of appearance, which is the generation order.
    paired_frames = []
    for c in items_df.cond.unique():
        if c == "clean":
            continue
        langs_c = set(items_df.loc[items_df.cond == c, "lang"])
        sub = items_df[items_df.lang.isin(langs_c)]
        paired_frames.append(paired_delta(sub, c, "clean").assign(contrast=f"{c}-clean"))
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(OUT / f"promptctl_paired_{OUT_TAG}.csv", index=False)

    print(f"\nwrote promptctl_{{items,agg,paired}}_{OUT_TAG}.csv to {OUT}")
    print("\n--- accuracy / response fidelity by condition (mean over languages) ---")
    # `aggregate` names the accuracy column `acc`, not `accuracy`.
    print(agg.groupby("cond")[["acc", "response_fidelity"]].mean().to_string())
    cols = ["lang", "n", "acc_a", "acc_b", "delta", "delta_lo", "delta_hi"]
    # Driven off the contrasts actually present, not off PIVOTS: under PC_FROM_ITEMS the env may
    # not match the run being re-analysed, and a silently skipped table is worse than a loud one.
    for contrast in [x for x in paired.contrast.unique() if x.endswith("_cot-clean")]:
        print(f"\n--- the comparison that matters: {contrast}, per language ---")
        print(paired[paired.contrast == contrast][cols].to_string(index=False))
    # Where the off-target responses actually GO is the finding, not a diagnostic: on 2026-08-24
    # 63% of en_cot's off-target output was Chinese, not English (bn 85%, te 95%, th 95%), which is
    # the same attractor the E9 interventions fall into. Print it so the next run cannot miss it.
    off = items_df[items_df.output_lang.astype(str) != items_df.lang]
    print("\n--- off-target destinations by condition (share of that arm's off-target rows) ---")
    for c, g in off.groupby("cond"):
        share = len(g) / max(1, len(items_df[items_df.cond == c]))
        top = g.output_lang.value_counts(normalize=True).head(4)
        print(f"  {c:22s} {share:5.1%} off-target -> "
              + "  ".join(f"{k}:{v:.0%}" for k, v in top.items()))


if __name__ == "__main__":
    main()
