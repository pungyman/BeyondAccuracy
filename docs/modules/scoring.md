# `scoring`

Source: [src/scoring.py](../../src/scoring.py)

Answer extraction / matching and output-language detection for **behavioral** experiments
(E10 and the cross-language validation E12), plus the **E5 benchmark scorers** (MGSM native
numeric, MC, cloze — see [Benchmark scorers](#benchmark-scorers-e5)). Promoted from
`scripts/e10_behavioral.py` so all experiments share one tested implementation.

> Matching is intentionally **lenient but applied identically** across all conditions, so the
> unbiased scorer noise cancels in paired comparisons (e.g. swap vs control).

`langdetect` is an **optional** dependency. If it is not installed, `detlang` degrades to
`"?"` and the script-based detectors still work.

## Answer extraction and matching

### `make_math_prompt(problem, instr_lang="ja") -> str`

Append a localized final-answer instruction to a problem. `\boxed{}` is language-neutral
notation; the surrounding prose is localized (`BOXED_INSTRUCTION` has `en` / `ja` / `zh`
entries) so, e.g., a Japanese prompt stays fully Japanese and does not prime English output.

### `extract_pred(text) -> str`

Pull the model's final answer, in priority order:

1. Last `\boxed{...}` match.
2. Cue phrases — `answer is` / `answer:` / `equals` / `结果是` / `答案是` / `答えは` / `=` —
   followed by a number.
3. Fallback: the last number in the text.

Returns `""` if nothing matches.

### `norm(s) -> str`

Normalize an answer for comparison: strip LaTeX wrappers (`\boxed`, `\text`, `\frac`, ...),
braces, spaces, commas, a trailing period; lowercase.

### `is_correct(pred, gold) -> bool`

`True` if `pred` matches `gold` exactly (after `norm`) **or** numerically within `1e-6`
(parsed via `fractions.Fraction`, so fractions compare correctly). Empty gold is always
`False`.

```python
from scoring import make_math_prompt, extract_pred, is_correct
prompt = make_math_prompt(problem, instr_lang="ja")
pred = extract_pred(model_output)
ok = is_correct(pred, gold)
```

## Benchmark scorers (E5)

The E5 benchmark harness ([`evaluate`](benchmarks.md#6-srcevaluatepy--the-reusable-harness)) needs
multiple-choice and cloze scoring, plus a language-robust numeric path for MGSM. These live
alongside the numeric path above.

### `extract_mgsm_answer(text, lang) -> str` — the native MGSM route

**Replaces the boxed-instruction approach for MGSM.** The native prompt scaffold
([`prompts.build_mgsm_prompt`](benchmarks.md#4-prompt-builders-srcpromptspy)) cues the model to end
with `<localized keyword>: <number>` using the vendored, cited `prompts.MGSM_ANSWER_PREFIX` table.
`extract_mgsm_answer` splits on that **same** keyword and takes the number after its last
occurrence (falling back to the last number anywhere). Crucially it first normalizes **native
digits** (Bengali `০–৯`, Telugu `౦–౯`, Thai `๐–๙`, … via `unicodedata.decimal`) to ASCII, so
`is_correct` compares numerically on non-Latin scripts (`bn te th`). `BOXED_INSTRUCTION` /
`make_math_prompt` are untouched and still serve the MATH extraction elsewhere.

### `extract_choice(text, n_choices=4) -> str` and `is_correct_mc(pred, gold) -> bool`

`extract_choice` pulls the model's chosen MC letter: a letter immediately after a localized
"Answer:" cue (`prompts.MMLU_ANSWER_CUE`, e.g. `答案: B`) wins; otherwise the last standalone
`A`–`D` (range set by `n_choices`); `""` if none. `is_correct_mc` is a case-insensitive exact
letter match (empty prediction is always wrong). MC is scored primarily by **log-likelihood** over
the option letters (`evaluate.score_continuations`); `extract_choice` backs the cheaper generative
secondary mode.

### `score_cloze(option_logprobs) -> str`

Argmax of the two XWinograd continuation log-probs → `"1"` / `"2"` (ties → `"1"`). XWinograd is
inherently log-likelihood: build the two full sentences
([`prompts.build_xwinograd_continuations`](benchmarks.md#4-prompt-builders-srcpromptspy)), score
each with `evaluate.score_continuations` (optionally length-normalized via its `char_norm` /
`token_norm` toggles), then `score_cloze`.

```python
from scoring import extract_mgsm_answer, extract_choice, is_correct_mc, score_cloze
pred = extract_mgsm_answer(output, "bn")          # native-digit aware
ok   = is_correct_mc(extract_choice(output), "B")
win  = score_cloze((lp_opt1, lp_opt2))            # "1" / "2"
```

## Output-language detection

### `script_dom(s) -> str`

Dominant **writing system** among alphabetic characters, by Unicode range: one of `Han`,
`Kana`, `Latin`, `Cyrillic`, `Bengali`, `Telugu`, `Thai`, `other`, or `none` (when there are
no alphabetic chars). Robust to digits/math/whitespace, which are ignored.

### `detlang(s) -> str`

`langdetect` ISO code with `zh-cn` / `zh-tw` folded to `zh`. Returns `"?"` if `langdetect` is
unavailable or unsure.

### `is_english(text) -> bool`

Coarse "did the output come out in English/Latin script?" signal: `script_dom(text) == "Latin"`.
Script-based, so robust to math notation — but it does not distinguish real English from
Latin-script gibberish.

### `english_word_hits(text) -> int`

Count of distinct **common English words** present (from a built-in `_COMMON_EN` set of
function/math words). A cheap fluency proxy: ~0 for gibberish, high for fluent prose.

### `coherent_english(text, min_hits=4) -> bool`

`True` only if the output is Latin script **and** reads as actual English
(`english_word_hits >= min_hits`). Stricter than `is_english`: rejects Latin-script gibberish
(e.g. a random-direction steer emitting `"Yuzo-123-000..."`), so it isolates *coherent*
English from a mere script flip.

- **Use as the primary "did injection produce English?" signal** in behavioral experiments.

```python
from scoring import script_dom, coherent_english, detlang
script_dom(output)        # e.g. "Han"
coherent_english(output)  # True only for genuine English prose
detlang(output)           # ISO code or "?"
```

## Related

- [intervene](intervene.md) — generate under a swap, then score the output here.
- [theory.md](../theory.md) — why the swap is expected to change the output language.
