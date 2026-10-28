# `geometry`

Source: [src/geometry.py](../../src/geometry.py)

**Geometry + statistics helpers** for the residual / subspace-comparison experiments
([shared_methods §5](../experiments/shared_methods.md)). Pure linear algebra and resampling —
**no model, no I/O** — so everything here is cheap and unit-tested
([tests/test_geometry.py](../../tests/test_geometry.py)).

Three families:

- **vector-vs-subspace** — `captured_fraction` (the E3 metric `ρ`).
- **subspace-vs-subspace** — `principal_angles`, `projector_distance`, `subspace_overlap` (E4 Phase 0).
- **bookkeeping / inference** — `split_problems`, `ordered_or_unordered_pairs`, `bootstrap_ci`.

All tensor ops run on `LINALG_DEVICE` (CPU/fp32). Subspace inputs are assumed **orthonormal**
`(d, r)` (everything from [candidates](candidates.md) / [subspace](subspace.md) is).

## Functions

### `split_problems(n, frac=0.5, seed=0) -> (train_idx, test_idx)`

Disjoint train/test split over **problem** indices `[0, n)` (`frac` = train fraction). Returns
sorted int arrays. Splitting on the problem index — not on rows — guarantees a problem's parallel
translations never straddle the split, so subspace estimation and `ρ` evaluation stay independent.

### `ordered_or_unordered_pairs(langs, ordered=False) -> list[(s, t)]`

Language pairs. `ordered=False` → `C(L, 2)` unordered (**55** for `L=11`); `ordered=True` →
`L(L-1)` ordered (**110**). The captured fraction is sign-invariant, so unordered loses nothing
for `ρ`; keep `ordered=True` only for directed analyses.

### `captured_fraction(V, M_s) -> Tensor`

The **captured-variance fraction** for each row of `V`:

\[ \rho(v) = \frac{\lVert M_s M_s^\top v\rVert^2}{\lVert v\rVert^2} = \frac{\lVert M_s^\top v\rVert^2}{\lVert v\rVert^2} \in [0,1]. \]

- **`V`** — `(m, d)` (e.g. translation vectors `h_t − h_s`); **`M_s`** — `(d, r)` orthonormal.
- **Returns** — `(m,)`. Because `M_s` is orthonormal, no `(d, d)` projector is materialized.
- Expected floor for an arbitrary `v` against a random rank-`r` subspace is `≈ r/d`.

### `principal_angles(Ma, Mb) -> Tensor`

Principal angles (radians, **ascending**) between `span(Ma)` and `span(Mb)`; `cos θ_i` are the
singular values of `Maᵀ Mb`. `0` = a shared direction, `π/2` = orthogonal.

### `projector_distance(Ma, Mb) -> float`

Basis-invariant subspace distance `‖P_a − P_b‖_F = √2 · ‖sin Θ‖₂`. `0` ⇒ identical span;
`√(2·rank)` ⇒ fully orthogonal subspaces.

### `subspace_overlap(Ma, Mb) -> float`

Captured-energy overlap `tr(P_a P_b)/r = ‖Maᵀ Mb‖_F² / r ∈ [0, 1]` — the fraction of `b`'s basis
captured by `a` (`r` = rank of `b`). `1` ⇒ same span.

### `bootstrap_ci(values, n_boot=1000, seed=0, ci=0.95) -> (mean, lo, hi)`

Percentile bootstrap CI for the **mean**, resampling the leading axis (problems). `values` is 1-D
(e.g. per-problem mean `ρ`). Returns `(mean, lo, hi)`.

```python
from geometry import (
    split_problems, ordered_or_unordered_pairs,
    captured_fraction, principal_angles, projector_distance, subspace_overlap, bootstrap_ci,
)

train_idx, test_idx = split_problems(cfg.n_samples, frac=0.5, seed=0)
# E3: how much of a held-out translation vector lives in M_s?
V   = acts["ja"][test_idx, layer, :] - acts["en"][test_idx, layer, :]   # (n_test, d)
rho = captured_fraction(V, M_s)                                         # (n_test,)
m, lo, hi = bootstrap_ci(rho.numpy())

# E4 Phase 0: are two subspaces the same?
dist    = projector_distance(M_alg1, M_diffpca)        # 0 = identical span
overlap = subspace_overlap(M_alg1, M_diffpca)          # 1 = identical span
```

## Related

- [candidates](candidates.md) — produces the subspaces these helpers compare.
- [shared_methods](../experiments/shared_methods.md) — the metric / protocol definitions.
- Notebooks: `04_E3_residuals.ipynb` (research-workspace reference) (`captured_fraction`),
  `05_E4_diffpca.ipynb` (research-workspace reference) (`principal_angles` /
  `projector_distance`).
- [tests/test_geometry.py](../../tests/test_geometry.py) — worked correctness checks.
