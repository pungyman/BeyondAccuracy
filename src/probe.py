from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def _to_np(X):
    if isinstance(X, Tensor):
        X = X.float().cpu().numpy()
    return np.asarray(X, dtype=np.float64)

def train_language_probe(X, y, seed: int = 0, test_size: float = 0.3):
    """11-way multinomial logistic probe for language ID.
    Returns (clf, test_accuracy, (X_test, y_test)). Stratified split, no leakage."""
    X = _to_np(X); y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr, ytr)
    return clf, float(clf.score(Xte, yte)), (Xte, yte)

def probe_accuracy(clf, X, y) -> float:
    return float(clf.score(_to_np(X), np.asarray(y)))

def pca_fit(X, k: int = 2, seed: int = 0) -> PCA:
    pca = PCA(n_components=k, random_state=seed)
    pca.fit(_to_np(X))
    return pca

from sklearn.model_selection import GroupKFold

def residualize_problem(X, problem_idx):
    """Partial out a categorical content covariate: subtract each problem-index group's mean
    activation (removes content shared across the parallel translations of the same problem).
    Returns residual np.float64 array, same shape as X."""
    X = _to_np(X); problem_idx = np.asarray(problem_idx)
    R = X.copy()
    for p in np.unique(problem_idx):
        m = problem_idx == p
        R[m] -= X[m].mean(axis=0, keepdims=True)
    return R


def unit_rows(X, eps: float = 1e-12):
    """L2-normalize each row of `X` (np.float64), zero-safe. Returns a new array.

    Used before variance-PCA fits so the objective matches the *angular*, per-vector captured
    fraction rho = ||Pv||^2/||v||^2 the experiments report. Raw PCA maximizes NORM-weighted
    captured variance, so under a heavy-tailed norm distribution (e.g. Qwen massive-activation
    dimensions ~1000x the typical scale) a handful of giant-norm rows dominate the fit and the
    unweighted mean rho collapses. Normalizing rows removes that leverage.
    """
    X = _to_np(X)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)

def leave_problems_out_probe(X, y, problem_idx, n_splits: int = 5, max_iter: int = 3000):
    """Group cross-validation by problem index (no problem's rows span train/test).
    Returns (mean_accuracy, per_fold_accuracies). Leakage-free language-decodability estimate."""
    X = _to_np(X); y = np.asarray(y); groups = np.asarray(problem_idx)
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=max_iter).fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    return float(np.mean(accs)), accs

def shuffled_label_floor(X, y, problem_idx, n_perms: int = 50, n_splits: int = 5, seed: int = 0):
    """Empirical chance level: permute language labels, run the leave-problems-out probe each time.
    Returns (mean, std) of permuted accuracies."""
    X = _to_np(X); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    scores = [leave_problems_out_probe(X, rng.permutation(y), problem_idx, n_splits=n_splits)[0]
              for _ in range(n_perms)]
    return float(np.mean(scores)), float(np.std(scores))
