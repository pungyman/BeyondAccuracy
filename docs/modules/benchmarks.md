# `benchmarks` (design spec — E5)

> **Status: P1 + P2 + P3 implemented.** The loaders ([`src/benchmarks.py`](../../src/benchmarks.py)),
> the localized prompt builders ([`src/prompts.py`](../../src/prompts.py)), the MC/cloze/native-MGSM
> scorers (extensions to [`scoring.py`](../../src/scoring.py)), and the reusable eval harness
> ([`src/evaluate.py`](../../src/evaluate.py)) exist and are exercised by
> `notebooks/06_E5_benchmarks.ipynb` (P1, clean baselines). **The intervention machinery of §5 —
> `InterventionSpec` / `build_layer_hooks` / `edit_mask` ([`src/intervene.py`](../../src/intervene.py)),
> the per-layer `M_s` builder ([`candidates.build_layer_subspaces`](../../src/candidates.py)), and
> the λ-schedule search ([`src/tuning.py`](../../src/tuning.py)) — ship (P2)** and are driven by
> `notebooks/07_E5_replication.ipynb`. Because P1 built the harness forward-compatible
> (`run_eval(..., hooks=...)` takes a `{layer: fn}` dict), P2 was a drop-in, not a rewrite. **P3 now
> ships the DiffPCA head-to-head:** the paired `DiffPCA − Alg1` per-language statistic
> ([`evaluate.paired_delta`](../../src/evaluate.py) / `low_resource_summary`, H2/H3) and the
> accuracy-vs-rank curve ([`tuning.rank_sweep`](../../src/tuning.py), H5) — both subspace-agnostic
> drop-ins on the P2 machinery — driven by `notebooks/08_E5_diffpca_headtohead.ipynb`. This page
> is the implementation companion to the E5 spec (research-workspace reference).

E5 needs three things the repo lacks: (1) loaders for the paper's three benchmarks, (2) scorers
for multiple-choice and cloze tasks (today [`scoring`](scoring.md) is MATH/numeric only), and (3)
a per-layer, scope-aware intervention driver plus a batched generate/score loop over
`model × benchmark × language × condition`. The subspace itself is unchanged — it is still built
from parallel MATH via [`candidates`](candidates.md) (see
E5 §2 (research-workspace reference)).

## 1. The three benchmarks

| Benchmark | HF dataset (config) | Task | Answer / scoring | Languages | ∩ our 11 |
|---|---|---|---|---|---|
| **MGSM** | `juletxara/mgsm` (per-lang subset; `test` = 250, `train` = 8 exemplars) | grade-school math | **numeric**, native few-shot CoT → `extract_mgsm_answer` (localized keyword + native-digit norm) | `bn de en es fr ja ru sw te th zh` (parallel) | **11** |
| **M-MMLU** | `CohereLabs/Global-MMLU` (per-lang; `test`) | knowledge QA | **4-way MC** (A/B/C/D) → log-likelihood over options | 42 langs incl. `bn de en es fr ja ru sw te zh` | **10** (no `th`) |
| **XWinograd** | `Muennighoff/xwinograd` (per-lang; `test`) | commonsense coreference | **2-way cloze** → log-likelihood over the two continuations | `en fr jp pt ru zh` (`jp`=Japanese) | **5** (`en fr ru zh ja`) |

**Language notes (accuracy-critical):**
- **MGSM** is fully parallel over our 11 and is the primary low-resource benchmark
  (E5 §3 (research-workspace reference)).
- **Global-MMLU** is the M-MMLU source of choice (covers 3/4 low-resource: `sw te bn`); **Thai is
  absent** — report on the 10-lang intersection. `openai/MMMLU` (professional translations, 14
  langs `AR BN DE ES FR HI ID IT JA KO PT SW YO ZH`) covers only 7 of ours (missing `en ru te
  th`) → keep it as a labeled robustness alternative, not primary.
- **XWinograd** uses subset name **`jp`** (not `ja`) for Japanese; `pt` is outside our 11 (no
  construction mean) so it is excluded by the eval-language policy. No low-resource coverage.

**Eval-language policy:** for each benchmark evaluate the **intersection of benchmark languages ∩
our 11 construction languages** (MGSM 11, M-MMLU 10, XWinograd 5). Report the set explicitly.

## 2. `src/benchmarks.py` — loaders and a unified schema

One record schema so the eval harness and scorers are benchmark-agnostic.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BenchItem:
    task: str                 # "mgsm" | "mmlu" | "xwinograd"
    lang: str                 # our canonical code (jp -> ja normalized)
    prompt: str               # fully built prompt (see prompt builders, §4)
    gold: str                 # numeric string | "A".."D" | "1"/"2"
    choices: tuple[str, ...] = ()   # MC/cloze continuations; () for generative
    meta: dict = field(default_factory=dict)  # id, subject, raw fields, few-shot used
```

Public surface:

| Function | Signature | Purpose |
|---|---|---|
| `benchmark_languages` | `(task: str) -> tuple[str, ...]` | languages the HF dataset provides |
| `eval_languages` | `(task: str, cfg: Config) -> tuple[str, ...]` | `benchmark_languages(task) ∩ cfg.langs` (the §1 policy) |
| `load_benchmark` | `(task, lang, split="test", *, cfg, n=None, seed=0, n_shot=0) -> list[BenchItem]` | load + build prompts; optional seeded subsample of size `n` (D6); `n_shot` uses MGSM `train` exemplars |
| `split_dev_test` | `(items, dev_frac=0.5, seed=0) -> (dev, test)` | per-benchmark **dev/test** split for lever tuning vs reporting (E5 §2 data roles); split on item id, seeded |

Loading maps each dataset's native fields into `BenchItem`:

- **MGSM** (`question`, `answer`/`answer_number`) → `gold` = numeric string; `choices=()`.
- **Global-MMLU** (`question`, `option_a..option_d`, `answer`) → `choices=(A,B,C,D)`, `gold` =
  the correct letter.
- **XWinograd** (`sentence`, `option1`, `option2`, `answer`∈{"1","2"}) → build the two full
  continuations by substituting each option into the `_`-blank of `sentence`; `gold` = "1"/"2".

Normalize XWinograd's `jp` → `ja` on load so downstream language handling is uniform. Datasets are
pulled via HF `datasets.load_dataset(...)`; add **`datasets`** to
[`pyproject.toml`](../../pyproject.toml) core deps. Cache locally under `data/benchmarks/`
(git-ignored like the other data), mirroring [`corpus`](corpus.md) conventions.

## 3. `scoring.py` extensions

Keep the existing numeric path ([`extract_pred`](scoring.md), [`is_correct`](scoring.md),
[`norm`](scoring.md)) for MGSM. Add:

### Multiple-choice (M-MMLU)
```python
def extract_choice(text: str, n_choices: int = 4) -> str
    # last standalone A/B/C/D (or localized "答案: B" cue); "" if none
def is_correct_mc(pred_letter: str, gold_letter: str) -> bool
```
MC accuracy has **two scoring modes**, both supported (report the primary, keep the other as a
robustness check):
- **Log-likelihood (primary, matches the paper / lighteval):** no generation — score each option
  under the model and pick argmax; robust to formatting and to small models that don't emit a
  clean letter. Uses the harness's `score_continuations` (§5).
- **Generative (secondary):** greedy-decode a short answer, `extract_choice`. Cheaper to eyeball,
  noisier on small/low-resource models.

### Cloze (XWinograd)
```python
def score_cloze(option_logprobs: tuple[float, float]) -> str   # "1"/"2" argmax
```
XWinograd is inherently log-likelihood: build the two full sentences, score each continuation's
tokens, pick the higher (optionally length-normalized — expose `LogProbCharNorm` /
`LogProbTokenNorm` toggles like lighteval, since normalization changes numbers).

### Per-language prompt/instruction text ([`src/prompts.py`](../../src/prompts.py))
The MGSM path uses the **native / attributable** route, *replacing* the boxed-instruction approach
for MGSM: rather than a hand-invented `\boxed{}` line in every language, MGSM reuses the dataset's
own localized `train` few-shot exemplars plus a **vendored, cited** localized answer keyword. The
two attributable literal tables live in `src/prompts.py`:
- **`MGSM_ANSWER_PREFIX: dict[str,str]`** — the localized answer keyword for all 11 (`en:"Answer"`,
  `de:"Antwort"`, `fr:"Réponse"`, `zh:"答案"`, `bn:"উত্তর"`, `te:"సమాధానం"`, `th:"คำตอบ"`, …). Source:
  the MGSM benchmark's canonical `LANG_TO_ANSWER_PREFIX` (Shi et al. 2022, arXiv:2210.03057; also
  in lighteval `translation_literal`). The prompt cues the model to end with `<keyword>: <number>`
  and [`scoring.extract_mgsm_answer`](scoring.md) splits the output on the *same* keyword — they
  stay in lockstep. **No machine-invented low-resource translations.**
- **`MMLU_ANSWER_CUE: dict[str,str]`** — the localized "Answer:" cue for the 10 MC langs.
- **XWinograd**: no instruction — it is pure continuation scoring.

Instruction language defaults to the **item's own language** (never prime English); it is recorded
in `BenchItem.meta["instr_lang"]` (with `n_shot`) so a fidelity analysis can condition on it.
[`BOXED_INSTRUCTION`](scoring.md) / `make_math_prompt` are **left untouched** (still used for MATH
extraction elsewhere); MGSM no longer depends on them.

## 4. Prompt builders ([`src/prompts.py`](../../src/prompts.py))

```python
def build_mgsm_prompt(item, exemplars=None, *, n_shot=0, instr_lang=None) -> str  # native few-shot, no \boxed{}
def build_mmlu_prompt(item, *, instr_lang=None) -> str                            # question + A-D + localized cue
def build_xwinograd_continuations(item) -> tuple[str, str]                        # (full_sentence_opt1, opt2)
```
`build_mgsm_prompt` concatenates `n_shot` localized exemplars (`question` + localized CoT `answer`)
then the test `question` with a trailing localized answer cue. The builders are **pure** (str/dict
in, str out — no model/tokenizer), so they are unit-testable offline; chat-templating happens later
in the harness via the same `tokenizer.apply_chat_template(..., add_generation_prompt=True)` path as
[`extract`](extract.md)/[`corpus`](corpus.md).

## 5. Modular intervention machinery

> **Implemented (P2).** `InterventionSpec`, `build_layer_hooks`, and `edit_mask` live in
> [`src/intervene.py`](../../src/intervene.py) and wrap the unchanged `ablate` primitive.
> [`evaluate.run_eval`](../../src/evaluate.py) applies `build_layer_hooks(spec)` via
> [`intervene.interventions`](intervene.md) exactly as the P1 seam anticipated (`hooks=None` =
> clean baseline); for the likelihood path a shared `intervene.MaskHolder` (passed as
> `run_eval(..., mask_holder=...)`) carries the prompt-only position scope. Band placement + λ
> strengths are chosen by [`tuning.coordinate_search`](../../src/tuning.py) on the dev split.

The reported scope is **prompt-only, mid-ablate / high-reinject** (E5 §4); the machinery makes
every axis a swappable parameter so all-steps and single-layer are one-line changes, not forks.
The primitive [`intervene.ablate(h, M_s, lam)`](intervene.md) is **unchanged**; the new code wraps
it.

```python
@dataclass(frozen=True)
class InterventionSpec:
    subspaces: dict[int, Tensor]     # layer_idx -> M_s (d, r)  (per-layer; from parallel MATH)
    lam: dict[int, float]            # layer_idx -> strength  (>0 ablate mid, <0 reinject high)
    scope: str = "prompt_only"       # "prompt_only" | "all_steps"
    # generative: prompt_only edits only the prefill pass.
    # likelihood: prompt_only edits only context positions (see edit_mask below).
```

- **`build_layer_hooks(spec) -> dict[int, callable]`** — produces `{layer: fn(h)->h}` for
  [`intervene.interventions`](intervene.md), each `fn` applying `ablate(h, spec.subspaces[layer],
  spec.lam[layer])`. The per-layer `lam` sign gives the mid-ablate / high-reinject bands (the edit
  is \(h - \lambda P h\), so \(\lambda<0\) amplifies language — re-injection).
- **Prompt-only under `generate`.** The hook must distinguish **prefill** (`output[0].shape[1] >
  1`) from **decode** (`shape[1] == 1`) and, for `scope="prompt_only"`, apply only during prefill.
  A stateful "prefill-done" flag per generation is the robust form (guards against chunked
  prefill). `scope="all_steps"` applies on every call (today's default behavior in
  [`interventions`](intervene.md), which the docstring already notes fires every step).
- **Likelihood tasks.** There is no decode loop; scope maps to a **position mask** over the single
  `[context + candidate]` forward pass: `prompt_only` edits only the context token positions,
  `all_steps` also edits the candidate positions. Implement as an optional `edit_mask` passed to
  the hook (boolean over `seq`), built from the known context length.

Band **placement** (which layers are "mid" vs "high") and the **strengths** are chosen by the
search in hyperparameter_search.md (research-workspace reference), seeded from each
model's E3 mid-network \(\rho\) trough.

## 6. `src/evaluate.py` — the reusable harness

One entry point that both the search (dev split) and the final report (test split) call.

```python
def run_eval(
    model, tok, items: list[BenchItem], *,
    hooks: dict[int, callable] | None = None,   # None = clean baseline; P2 passes build_layer_hooks(spec)
    batch_size: int = 16, max_new_tokens: int = 512, device=None,
    max_batch_tokens: int | None = 4096, model_name: str | None = None,
    mask_holder=None,                            # intervene.MaskHolder for likelihood prompt-only scope
    cond: str | None = None,                     # row label, e.g. "alg1_tuned" / "alg1_paperfixed"
) -> "pd.DataFrame":
    """One row per item: model, task, lang, id, pred, gold, correct(bool), output_script, output_lang, cond."""
```

The `hooks` argument is the intervention seam (see §5): `None` = clean baseline (P1); P2 passes
`build_layer_hooks(spec)`, applied via `intervene.interventions(model, hooks)`. `mask_holder` (built
from the same spec) carries the likelihood prompt-only position scope; `cond` overrides the
condition label so paired before/after rows are taggable.

- **Generative path (MGSM):** length-bucketed greedy `model.generate` (reusing the batching in
  [`extract`](extract.md)) under the hook context, then
  `is_correct(extract_mgsm_answer(out, lang), gold)`; also records `scoring.script_dom` / `detlang`
  for the **fidelity** axis (E5 §7).
- **Likelihood path (M-MMLU, XWinograd):** `score_continuations(model, tok, context, continuations)`
  runs a teacher-forced forward per option and sums the continuation token log-probs (with
  `char_norm` / `token_norm` toggles), argmax → correctness. No sampling. Prompt-only position
  scoping (P2) is carried by a `mask_holder` set from the context length before each forward, so an
  intervention edits only context tokens; the scored continuation tokens pass through untouched.
- **Aggregation:** `aggregate(df)` reduces the per-item frame to per-(model, task, language) mean
  accuracy with `geometry.bootstrap_ci`; `write_results(...)` emits **model-stamped** tidy CSVs
  (`results/E5/<model>/E5_phase1_clean_<model>.csv` per item and `..._agg_<model>.csv`), matching the E3/E4
  convention. Paired Alg1-vs-DiffPCA deltas and figures come with P2–P4.

## 7. Testing

Mirror the existing `tests/` discipline (`test_candidates.py`, `test_intervene.py`):
- **`scoring`**: unit tests for `extract_choice`, `is_correct_mc`, `score_cloze`, and the numeric
  path on non-Latin scripts (`bn te th` numerals) — the current `scoring` has *no* tests despite
  being called "tested", so add them.
- **`benchmarks`**: schema/coverage tests with a tiny fixture (offline); assert
  `eval_languages` = the §1 intersections; assert XWinograd `jp→ja` normalization.
- **intervention machinery**: `build_layer_hooks` produces the right per-layer `lam` signs;
  prompt-only applies on prefill only (assert a decode-step tensor is untouched); `edit_mask`
  restricts edits to context positions.
- Keep dataset-dependent tests behind a `skipif` (like `test_corpus.py`) so CI stays offline.

## Related
- E5 spec (research-workspace reference) — the experiment this serves.
- hyperparameter_search.md (research-workspace reference) — how `InterventionSpec`'s
  `lam` / rank / angular are chosen without overfitting the reported split.
- [candidates](candidates.md) — builds the per-layer `M_s` (Alg1 / DiffPCA) from parallel MATH.
- [intervene](intervene.md) — the `ablate` primitive and `interventions` hook the machinery wraps.
- [scoring](scoring.md) — the numeric path reused for MGSM; extended here for MC/cloze.
