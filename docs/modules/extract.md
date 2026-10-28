# `extract`

Source: [src/extract.py](../../src/extract.py)

Loads the model/tokenizer and extracts the **final-token residual stream at every layer**.
These activations are the raw material for building the language subspace
([subspace](subspace.md)) and for the probe ([probe](probe.md)).

## Functions

### `load_model_and_tokenizer(cfg, device)`

Load a causal LM and its tokenizer per the [config](config.md).

- **`cfg`** — a `Config` (uses `cfg.model_name`, `cfg.dtype`).
- **`device`** — target device (`device_map=device`), typically `get_device()`.
- **Returns** — `(model, tokenizer)`, with `model` in `.eval()` mode.

Key tokenizer setup:

- **`padding_side = "left"`** — so the **last token is always real**, never a pad token. The
  whole extraction reads the final position, so left-padding is essential for correctness.
- **`pad_token = eos_token`** if the tokenizer has no pad token, so batched padding works.

```python
from config import Config, get_device
from extract import load_model_and_tokenizer

cfg = Config()
model, tok = load_model_and_tokenizer(cfg, get_device())
```

### `extract_last_token_per_layer(model, tok, prompts, device, batch_size=16) -> Tensor`

Decorated with `@torch.no_grad()`. For each prompt, run a forward pass with
`output_hidden_states=True` and collect the **last-token** hidden state at **every block
output**.

- **`prompts`** — list of raw prompt strings (the chat template is applied internally, so pass
  raw prompts, not pre-templated ones).
- **`batch_size`** — prompts per forward pass (default 16).
- **Returns** — a tensor of shape **`(n_prompts, n_layers, d_model)`**, in **fp32 on CPU**.

How it works, line by line:

- Prompts are chat-templated (`apply_chat_template(..., add_generation_prompt=True)`),
  tokenized with left-padding, and run with `use_cache=False`.
- `hidden_states` from the model is a tuple of length `n_layers + 1`. The code uses
  **`hidden_states[1:]`** — dropping index 0, which is the **embedding layer output** — so
  index `k` of the result is the **output of decoder block `k`**
  (`hidden_states[k+1] = output of block k`).
- It stacks the layers, selects the **last position** (`[:, :, -1, :]` — real thanks to
  left-padding), transposes to `(batch, layers, d)`, and moves to `float().cpu()`.

> Output is fp32/CPU regardless of the model's compute dtype, so downstream linear algebra
> ([subspace](subspace.md)) is numerically stable on `LINALG_DEVICE`.

### `acts_cache_path(cfg, cache_dir=None) -> Path`

Where per-language activations for `cfg` are cached:
`cache_dir / f"acts_{model}_n{n_samples}_{sample}-s{seed}.pt"` (model = the part of
`cfg.model_name` after the last `/`). The sampling scheme + seed are part of the filename so a
stratified cache is never confused with a legacy first-`n` one (e.g.
`acts_Qwen2.5-1.5B-Instruct_n512_stratified-s0.pt`); pre-existing `acts_<model>_n<N>.pt` files no
longer match and are regenerated. `cache_dir` defaults to `config.ACTS_DIR`
(`data/activations`).

### `acts_meta_path(cfg, cache_dir=None) -> Path` · `load_acts_meta(cfg, cache_dir=None) -> dict`

The activation cache has a **sidecar** `…​.meta.json` recording the run's provenance and the
exact problem selection: `{model, n, sample, seed, strata_keys, indices, labels}`. `indices`
are the chosen dataset rows (sorted) and `labels` are their aligned metadata
(`level` / `type` / `outcome`). `load_acts_meta` reads it — used to recover the selection for
reproducible splits and for the E4 content probe (which predicts `type`).

### `load_or_extract_acts(cfg, device=None, batch_size=16, force=False, cache_dir=None, allow_download=True) -> dict`

Convenience wrapper that returns per-language final-token activations
**`{lang: (n_samples, n_layers, d)}`**, cached on disk.

- If `acts_cache_path(cfg, cache_dir)` exists (and not `force`), it is loaded — instant, no model.
- Otherwise, if `allow_download=True`, it tries to download the matching `.pt` and `.meta.json`
  from `cfg.acts_repo` (default empty; set `LRD_ACTS_REPO` explicitly to use a remote cache) into `data/activations/`.
- Only when both local and remote caches miss does it load the model (`load_model_and_tokenizer`,
  above), select the `cfg.n_samples` problems once via
  [`corpus.select_problem_indices`](corpus.md) (stratified by default), load the parallel corpus
  for those indices, extract every language in `cfg.langs`, cache the dict **and write the
  `…​.meta.json` sidecar**, then return it.
- The model forward pass is the **only** non-CPU work in the E3 / E4-Phase-0 pipeline; subsequent
  runs reuse the cache.

```python
from config import Config, get_device
from extract import load_or_extract_acts

cfg = Config(n_samples=150)
acts = load_or_extract_acts(cfg, device=get_device())
# acts["en"].shape == (150, n_layers, d)
```

> Requires read access to the activation repo or the parallel corpus on disk (see
> [corpus](corpus.md), `data/README.md`, and `data/activations/README.md`). To skip extraction
> entirely, place a precomputed `.pt` and `.meta.json` pair in `data/activations/`.

### `upload_acts(cfg, cache_dir=None, repo_id=None, private=True) -> None`

Uploads the local activation tensor and sidecar to a Hugging Face dataset repo. This is the
programmatic equivalent of:

```bash
python scripts/sync_acts.py push --model Qwen/Qwen2.5-7B-Instruct --n 512
```

## Putting it together

```python
acts = {                                              # lang -> (n_samples, n_layers, d)
    lang: extract_last_token_per_layer(model, tok, prompts[lang], device)
    for lang in cfg.langs
}
# Select one layer to analyze:
layer = 12
acts_by_lang = {lang: a[:, layer, :] for lang, a in acts.items()}   # lang -> (n_samples, d)
```

## Related

- [corpus](corpus.md) — produces the `prompts` lists.
- [subspace](subspace.md) — consumes `acts_by_lang` (a single layer) to build `W` and `M_s`.
- [config](config.md) — `dtype`, `model_name`, device selection.
