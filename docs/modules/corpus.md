# `corpus`

Source: [src/corpus.py](../../src/corpus.py)

Loads the parallel multilingual prompts and formats them for the model. This is the first
step of the pipeline — it produces the index-aligned prompts that [extract](extract.md) turns
into activations.

## Why "parallel"

Each language file in `DATA_DIR` ([config](config.md)) is a JSON list of the *same* problems,
translated. Selecting the same record indices from each language guarantees that row `i` is the
same problem across all languages, so any activation difference at row `i` is due to language,
not content. See [theory.md](../theory.md).

## Sampling: which `n` problems?

The MATH corpus is skewed (≈31% Level 5; 36 `level × type` strata), so the legacy first-`n`
slice would over/under-represent difficulty levels and subjects. The default is therefore a
**proportional stratified** sample (decision
D4 (research-workspace reference)): each `level × type` stratum keeps its corpus share via the
largest-remainder rule, drawn with a fixed seed. `Config.sample` (`"stratified"` | `"first"`),
`Config.strata_keys`, and `Config.seed` control it. The same indices are used for every
language, and [extract.load_or_extract_acts](extract.md) persists them in a sidecar so the
selection is reproducible.

## Functions

### `stratified_indices(labels, n, seed=0) -> np.ndarray`

Proportional (largest-remainder) stratified sample of `n` indices into `labels`. Strata are the
distinct values of `labels` (hashable, e.g. `(level, type)` tuples), iterated in a stable sorted
order. Each stratum gets `round_lr(n · s_k / N)` slots (floors plus the leftover handed to the
largest fractional remainders, summing to exactly `n`); members are drawn without replacement.
Returns a **sorted** int array. Raises `ValueError` if `n` is outside `[0, N]`.

### `select_problem_indices(records, n, sample="stratified", seed=0, strata_keys=("level","type")) -> (indices, labels)`

Choose `n` problem indices and return them with aligned metadata. `sample="stratified"` uses
`stratified_indices` over `strata_keys`; `sample="first"` returns `arange(n)` (legacy). `labels`
is the per-selected-row metadata (`strata_keys` + `outcome`), ordered to match `indices`.

### `load_parallel(langs, n, prompt_key="translated_prompt", cfg=None, *, indices=None) -> dict[str, list[str]]`

Load `n` index-aligned prompts per language.

- **`langs`** — iterable of language codes (e.g. `cfg.langs`). Each must have a
  `DATA_DIR/{lang}.json` file.
- **`n`** — number of prompts per language.
- **`prompt_key`** — JSON field to extract from each record (default `"translated_prompt"`).
- **`cfg`** — supplies `sample` / `strata_keys` / `seed` when `indices` is not given.
- **`indices`** — optional precomputed selection (e.g. from `select_problem_indices` or a
  persisted sidecar); bypasses sampling so a run reuses an exact selection.
- **Returns** — `{lang: [prompt_str, ...]}`, each list of length `n`, index-aligned across
  languages.
- **Raises** — `AssertionError` if the language files have differing lengths (a non-parallel
  corpus), guarding against misaligned data.

```python
from config import Config
from corpus import load_parallel

cfg = Config()                       # sample="stratified" by default
prompts = load_parallel(cfg.langs, n=cfg.n_samples, cfg=cfg)
prompts["en"][0]   # a sampled English problem
prompts["ja"][0]   # the same problem, in Japanese
```

### `build_chat_inputs(tokenizer, prompts) -> list[str]`

Apply the tokenizer's default chat template to each prompt, as a `user` turn with
`add_generation_prompt=True` and no system override. Returns the rendered template **strings**
(`tokenize=False`), not token ids.

- **`tokenizer`** — a HuggingFace tokenizer with `apply_chat_template`.
- **`prompts`** — list of raw prompt strings.
- **Returns** — list of chat-formatted strings, ready to tokenize.

> [extract.extract_last_token_per_layer](extract.md) applies the same chat template
> internally, so you only need `build_chat_inputs` when you tokenize/generate manually.

## Related

- [config](config.md) — supplies `DATA_DIR`, `langs`, `n_samples`, `prompt_key`.
- [extract](extract.md) — consumes these prompts to produce activations.
