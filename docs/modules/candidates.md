# `candidates`

Source: [src/candidates.py](../../src/candidates.py)

The **five comparable rank-`r` subspaces at a single layer** — one source of truth shared by the
residual sweep (E3 (research-workspace reference)) and the geometry pre-flight
(E4 Phase 0 (research-workspace reference)). All five are built from the **same**
per-language activations at one layer, so they are directly comparable via
[geometry](geometry.md) and drop-in compatible with [intervene](intervene.md) / [probe](probe.md).

Four of them are **one PCA, indexed by (population × centering)** (see the unification in
[theory.md](../theory.md)); `random` is the dimensionality null:

| name | what it is | population × centering | module |
|------|------------|------------------------|--------|
| `alg1`    | two-step SVD, refined anchor      | language means, learned anchor   | [subspace](subspace.md) `build_language_subspace` |
| `none`    | PCA of centered means             | language means, uniform centroid | [subspace](subspace.md) `build_subspace_variant(mode="none")` |
| `diffpca` | PCA of within-problem residuals   | instances, per-problem mean      | this module |
| `pooled`  | top-variance / content directions | instances, pooled centroid       | [controls](controls.md) `pooled_pca_subspace` |
| `random`  | Haar-random rank-`r`              | — (null)                         | [controls](controls.md) `random_orthonormal_subspace` |

## Functions

### `pooled_long_form(acts_by_lang) -> (X, lang_idx, problem_idx)`

Stack per-language `(n, d)` activations into pooled **long form**.

- **`acts_by_lang`** — `{lang: (n, d)}` at one layer (must be parallel: equal `n`).
- **Returns**
  - `X` — `(nL, d)` in **language-major** order (all of language 0, then language 1, …).
  - `lang_idx` — `(nL,)` `= [0]*n + [1]*n + …`.
  - `problem_idx` — `(nL,)` `= [0..n-1]` tiled `L` times.

The `problem_idx` tiling is the whole point: it is exactly the grouping
[`probe.residualize_problem`](probe.md) partials out, so each problem's `L` parallel
translations share a `problem_idx` and their common **content** is subtracted.

### `diffpca_subspace(acts_by_lang, rank, *, angular=True) -> (M_s, explained_variance_ratio_)`

**DiffPCA**: the top-`rank` principal components of the within-problem-residualized activations
(the metric-optimal, content-invariant subspace of
[shared_methods §4.3](../experiments/shared_methods.md)).

- Residualizes via `pooled_long_form` → `residualize_problem` (content removed), then PCA.
- **`angular`** — PCA objective on the residual rows:
  - `True` (default, **shipped**) fits on L2-normalized rows ([`probe.unit_rows`](probe.md)) so the
    objective matches the scale-free captured-fraction ρ; the **massive-activation guardrail**
    (Qwen 7B+), see E3_results §methods addendum (research-workspace reference).
  - `False` fits on **raw** residuals (norm-weighted variance PCA). Kept as an explicit option for
    the E4 Phase-0 angular-vs-raw geometry reconciliation; not a default on a flagged model.
- **Returns** `(M_s, scree)` where `M_s` is `(d, rank)` orthonormal and `scree` is the length-`rank`
  `explained_variance_ratio_`. The **tail** of the scree is the entry point for the
  intrinsic-dimensionality / rank-`>10` question (E4 Phase 4): unlike `alg1`, DiffPCA is **not**
  structurally capped at `L-1`.

### `build_candidate_subspaces(acts_by_lang, rank, seed=0, include=CANDIDATE_NAMES, diffpca_angular=True) -> (subspaces, meta)`

Build the requested subspaces from one layer's per-language activations.

- **`include`** — subset of `CANDIDATE_NAMES = ("alg1", "none", "diffpca", "pooled", "random")`.
- **`diffpca_angular`** — forwarded to `diffpca_subspace` (`True` = shipped angular fit; `False` =
  raw variance PCA).
- **Returns**
  - `subspaces` — `{name: (d, rank)}` orthonormal.
  - `meta` — side information; currently `meta["diffpca_scree"]` (present only if `"diffpca"` was
    requested).

```python
from candidates import build_candidate_subspaces, CANDIDATE_NAMES

acts_by_lang = {lang: acts[lang][:, layer, :] for lang in cfg.langs}   # {lang: (n, d)}
subs, meta = build_candidate_subspaces(acts_by_lang, rank=cfg.rank, seed=0)
M_diffpca = subs["diffpca"]               # (d, rank)
scree     = meta["diffpca_scree"]         # (rank,) explained-variance ratios
```

> **Honest estimation.** Build the subspaces on a **train** split of problems
> (`geometry.split_problems`) and evaluate on held-out translation vectors — the notebooks pass
> `acts_by_lang` restricted to `train_idx`.

## Related

- [geometry](geometry.md) — compares these subspaces (`captured_fraction`, `principal_angles`,
  `projector_distance`).
- [subspace](subspace.md) / [controls](controls.md) — provide `alg1`/`none` and `pooled`/`random`.
- [probe](probe.md) — `residualize_problem`, the content-removal step behind DiffPCA.
- [theory.md](../theory.md) — the **(population × centering)** unification of the four PCA variants.
- Notebooks: `04_E3_residuals.ipynb` (research-workspace reference),
  `05_E4_diffpca.ipynb` (research-workspace reference).
