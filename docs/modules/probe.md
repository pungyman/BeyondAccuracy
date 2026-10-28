# `probe`

Source: [src/probe.py](../../src/probe.py)

A multinomial logistic probe for **language ID** plus leakage-free evaluation tools. The probe
measures *how decodable language is* from activations — typically compared before vs after an
intervention ([intervene](intervene.md)), or against the control subspaces
([controls](controls.md)).

## The leakage problem

Parallel translations of the same problem share content. A naive random train/test split lets
the probe memorize problems and inflate accuracy. The tools below address this by splitting or
residualizing **by problem index**. See [theory.md](../theory.md).

All inputs `X` may be torch tensors, numpy arrays, or lists — the private `_to_np` helper
coerces to `np.float64`.

## Functions

### `train_language_probe(X, y, seed=0, test_size=0.3)`

Train an 11-way multinomial logistic-regression probe with a stratified split (no leakage
control beyond stratification).

- **`X`** — `(n_samples, d)` activations.
- **`y`** — language labels.
- **Returns** — `(clf, test_accuracy, (X_test, y_test))`. `clf` is a fitted
  `LogisticRegression(max_iter=3000, C=1.0)`.

```python
from probe import train_language_probe
clf, acc, (Xte, yte) = train_language_probe(X, y)
```

### `probe_accuracy(clf, X, y) -> float`

Score a fitted probe on new data: `clf.score(X, y)`.

### `pca_fit(X, k=2, seed=0) -> PCA`

Fit an sklearn `PCA` with `k` components (default 2). Used for visualizing language structure.

### `residualize_problem(X, problem_idx) -> np.ndarray`

Partial out a categorical **content** covariate: subtract each problem-index group's mean
activation, removing the content shared across the parallel translations of one problem.

- **Returns** — residual array, same shape as `X` (`np.float64`). What remains is closer to
  pure language variation.

### `leave_problems_out_probe(X, y, problem_idx, n_splits=5, max_iter=3000)`

Group cross-validation by problem index (`sklearn.GroupKFold`), so no problem's rows span the
train/test boundary — a leakage-free estimate of language decodability.

- **Returns** — `(mean_accuracy, per_fold_accuracies)`.

### `shuffled_label_floor(X, y, problem_idx, n_perms=50, n_splits=5, seed=0)`

Empirical **chance level**: permute the language labels and run `leave_problems_out_probe`
each time.

- **Returns** — `(mean, std)` of the permuted accuracies. Real decodability is meaningful only
  if it sits well above this floor.

```python
from probe import (
    leave_problems_out_probe, residualize_problem, shuffled_label_floor,
)
R = residualize_problem(X, problem_idx)              # remove content
acc, folds = leave_problems_out_probe(R, y, problem_idx)
floor_mean, floor_std = shuffled_label_floor(R, y, problem_idx)
```

## Related

- [theory.md](../theory.md) — why leakage control matters here.
- [intervene](intervene.md) — probe before/after ablation to test if `M_s` carried the
  language signal.
- [controls](controls.md) — probe on activations edited by null subspaces for comparison.
