# `config`

Source: [src/config.py](../../src/config.py)

Shared configuration for the whole pipeline: the experiment `Config` dataclass, repo paths,
the canonical language list, seeding, and device selection. Every other module reads its
defaults from here.

## Paths

Resolved relative to the package location (`Path(__file__).resolve().parents[1]` = the repo
root):

| Name | Value | Purpose |
|------|-------|---------|
| `REPO` | repo root | Repo root. |
| `DATA_DIR` | `REPO/data/parallel_multilingual_math` | Per-language prompt JSONs read by [corpus](corpus.md). |
| `ACTS_DIR` | `REPO/data/activations` | Cached activation tensors and sidecar metadata. |
| `RESULTS_DIR` | `REPO/results` | Where experiment outputs are written. |
| `FIGS_DIR` | `REPO/figs` | Where figures are written. |

## `LANGS`

```python
LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
```

The 11 languages, alphabetical, matching notebook 01. Order is significant: it fixes the
column order of the mean matrix `W` and the label ordering for the probe.

## `Config`

A frozen dataclass holding all experiment knobs.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `model_name` | `str` | `"Qwen/Qwen2.5-1.5B-Instruct"` | HuggingFace model id. |
| `data_repo` | `str` | `""` (local files by default; override with `LRD_DATA_REPO`) | HF dataset repo for prompt JSON fallback. |
| `data_revision` | `str` | `"main"` | HF revision for prompt JSON fallback. |
| `acts_repo` | `str` | `""` (no remote cache by default; override with `LRD_ACTS_REPO`) | HF dataset repo for cached activation fallback/upload. |
| `acts_revision` | `str` | `"main"` | HF revision for activation cache fallback. |
| `langs` | `tuple[str, ...]` | `tuple(LANGS)` | Languages to use. |
| `n_samples` | `int` | `32` | Parallel prompts per language. |
| `prompt_key` | `str` | `"translated_prompt"` | JSON field read from each language file. |
| `seed` | `int` | `0` | Global seed. |
| `sample` | `str` | `"stratified"` | Problem sampling mode (`"stratified"` or legacy `"first"`). |
| `strata_keys` | `tuple[str, ...]` | `("level", "type")` | Metadata fields used by stratified sampling. |
| `intervene_layer` | `int \| None` | `None` | Layer for subspace/intervention; `None` resolves to `n_layers // 2` **at runtime** (not stored). |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Model compute dtype. |

### Properties

- **`n_langs -> int`** — `len(self.langs)`.
- **`rank -> int`** — `n_langs - 1`. The default subspace rank: with 11 languages the
  language means span at most an \(L{-}1\) dimensional affine subspace after removing the
  shared anchor, so `rank = 10`. See [theory.md](../theory.md).

```python
from config import Config
cfg = Config()
cfg.n_langs   # 11
cfg.rank      # 10
```

> `intervene_layer` is intentionally `None` by default. Callers resolve it as
> `cfg.intervene_layer if cfg.intervene_layer is not None else n_layers // 2` once the model's
> layer count is known (see [workflow.md](../workflow.md)).

## Functions

### `set_seeds(seed: int = 0) -> None`

Seed `random`, `numpy`, and `torch` together for reproducibility.

### `get_device() -> torch.device`

Pick the best available **compute** device, in order: MPS, then CUDA, then CPU. Used for the
model forward pass.

## `LINALG_DEVICE`

```python
LINALG_DEVICE = torch.device("cpu")   # SVD on MPS is unreliable
```

A separate, fixed CPU device used by all linear algebra ([subspace](subspace.md),
[controls](controls.md)). The split — forward pass on `get_device()`, SVD on
`LINALG_DEVICE` — exists because `torch.linalg.svd` produces unreliable results on MPS.

## Related

- [corpus](corpus.md) reads `DATA_DIR` / `prompt_key`.
- [extract](extract.md) reads and writes activation caches under `ACTS_DIR`.
- [subspace](subspace.md) and [controls](controls.md) compute on `LINALG_DEVICE`.
