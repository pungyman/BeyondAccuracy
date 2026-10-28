# `controls`

Source: [src/controls.py](../../src/controls.py)

Null / baseline subspaces to compare against the real language subspace `M_s`
([subspace](subspace.md)). Each one isolates a different confound, so an effect that survives
all three is attributable to the language structure rather than to dimensionality, variance,
or the construction itself. See the control logic in [theory.md](../theory.md).

All functions return an orthonormal-ish `(d, rank)` basis in fp32/CPU (`LINALG_DEVICE`),
drop-in compatible with [intervene](intervene.md) and [probe](probe.md) wherever `M_s` is
used.

## Functions

### `random_orthonormal_subspace(d, rank, seed) -> Tensor`

A **Haar-random** orthonormal `(d, rank)` subspace: QR of a Gaussian matrix, `Q[:, :rank]`.

- A generic rank-`r` direction set with no language structure.
- **Falsifies**: "any rank-`r` subspace would produce this effect." If editing along random
  directions matches editing along `M_s`, the effect is about dimensionality, not language.

### `pooled_pca_subspace(acts_by_lang, rank) -> Tensor`

Top-`rank` principal directions of the **pooled** (language-collapsed) activations: concatenate
all languages, center, SVD, take `Vh[:rank].T`.

- Generic **high-variance content** directions, language-agnostic (languages are mixed before
  PCA, so language identity is not what drives these directions).
- **Falsifies**: "the effect is just from removing high-variance directions." If pooled-PCA
  matches `M_s`, variance — not language — explains it.

### `permuted_label_subspace(acts_by_lang, rank, seed) -> Tensor`

A **permutation null** for "`M_s` encodes language": shuffle which language each sample belongs
to (preserving group sizes), recompute per-group means, and rebuild via the same two-step SVD
(`language_means` + `build_language_subspace`).

- **Falsifies**: "the two-step SVD would find a 'language' subspace even from meaningless
  labels." If the permuted subspace behaves like `M_s`, the construction is finding structure
  in noise.

```python
from controls import (
    random_orthonormal_subspace, pooled_pca_subspace, permuted_label_subspace,
)
d = next(iter(acts_by_lang.values())).shape[1]
M_rand = random_orthonormal_subspace(d, rank=10, seed=0)
M_pca  = pooled_pca_subspace(acts_by_lang, rank=10)
M_perm = permuted_label_subspace(acts_by_lang, rank=10, seed=0)
```

## Related

- [subspace](subspace.md) — the real `M_s` these baselines are compared against.
- [theory.md](../theory.md) — what hypothesis each control falsifies.
- [probe](probe.md) / [intervene](intervene.md) — run the same analysis on each subspace.
