"""E6 Plan-B affine intervention estimators (docs/experiments/E6_planb_operators.md).

Alg1/diffPCA fix the *operator* (hard orthogonal projection of a rank-r span, scalar lam) and
vary only how the span is estimated. This module varies the operator instead: every estimator
below returns a per-layer `intervene.AffineEdit` — a factored affine map `h <- h - lam*delta(h)`
— derived in closed form from the same single-layer `{lang: (n, d)}` view that
`candidates.build_candidate_subspaces` consumes. Three families:

- **erasure** — `leace_edit`: oblique (covariance-whitened) projection of the language-mean
  span; the minimal-MSE edit after which no linear probe can decode language
  (Belrose et al. 2023, arXiv 2306.03819). The whitening also neutralizes massive-activation
  coordinates, the failure mode of raw/angular PCA at 7B (E3 results, E6 Phase 0).
- **estimation** — `wiener_edit`: under the random-effects model `h = c_i + g_l + q + eps`,
  remove the posterior mean of the language effect, `E[g|h] = Sigma_g (Sigma_tot)^-1 (h - mu)`.
  Derives the per-direction attenuation profile that the E5 search approximates with one
  scalar lam.
- **transport** — `meanshift_edit` ⊂ `adain_edit` ⊂ `gaussian_ot_edit` ⊂ `translator_edit`:
  per-source-language affine maps onto a target distribution (the cross-language barycenter =
  the ablate analogue, or a pivot/target language = the swap analogue). The translator rung is
  fit on *problem-matched pairs* — supervision only a parallel corpus can provide.

Plus `uniform_shrink_edit` (the energy-matched nonspecific control) and `projection_edit`
(wraps an existing `(d, r)` subspace so alg1/diffpca run through the identical hook path).

Estimation conventions match candidates.py: fit on the full parallel construction corpus (the
benchmark-disjoint MATH cache — disjointness, not an internal split, protects the eval),
numerics in numpy float64 on CPU, factors returned as fp32 tensors. Full-covariance estimators
(`leace`, `wiener`, `gaussian_ot`) do one or more `d x d` eigendecompositions per layer —
seconds at d=1536, ~10-30 s at d=3584 — so build them for the intervention band, not all layers,
when iterating.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor

from subspace import build_language_subspace, language_means
from candidates import build_layer_subspaces, pooled_long_form
from probe import residualize_problem
from intervene import AffineEdit

#: language-agnostic edits (one edit serves every eval language)
EDIT_NAMES: tuple[str, ...] = ("leace", "wiener", "wdiffpca", "shrink")
#: language-conditional transports (need `source=`, one edit per eval language)
TRANSPORT_NAMES: tuple[str, ...] = ("meanshift", "adain", "gaussian_ot", "translator")


# --- shared numpy helpers ------------------------------------------------------------------------

def _pooled_np(acts_by_lang: dict[str, Tensor]):
    """`(X, lang_idx, problem_idx, langs)` — pooled float64 rows + labels (language-major)."""
    X, lang_idx, problem_idx = pooled_long_form(acts_by_lang)
    return (X.numpy().astype(np.float64), lang_idx.numpy(), problem_idx.numpy(),
            list(acts_by_lang))


def _shrunk(S: np.ndarray, shrink: float) -> np.ndarray:
    """Ledoit-Wolf-style ridge toward the scaled identity: (1-g)S + g (tr S / d) I."""
    d = S.shape[0]
    return (1.0 - shrink) * S + shrink * (np.trace(S) / d) * np.eye(d)


def _sqrt_invsqrt(S: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric PSD square root and inverse square root via eigh (eigenvalues floored)."""
    evals, evecs = np.linalg.eigh(S)
    evals = np.maximum(evals, floor)
    return ((evecs * np.sqrt(evals)) @ evecs.T,
            (evecs / np.sqrt(evals)) @ evecs.T)


def _inv_shrunk(S: np.ndarray, shrink: float, floor: float = 1e-10) -> np.ndarray:
    """Inverse of `_shrunk(S, shrink)` via eigh with floored eigenvalues (the Wiener denominator)."""
    evals, evecs = np.linalg.eigh(_shrunk(S, shrink))
    return (evecs / np.maximum(evals, floor)) @ evecs.T


def layer_whitener(acts_by_lang: dict[str, Tensor], shrink: float = 0.1,
                   *, need: tuple[str, ...] = ("leace", "wiener", "wiener_marginal")) -> dict:
    """One layer's rank- and basis-independent second-moment work, for reuse across a sweep.

    `leace_edit` / `wiener_edit` accept the result as `whiten=`. Everything here is a function of
    the layer's activations and `shrink` ALONE — not of `rank`, not of the injected `basis` — so a
    caller sweeping `(basis, operator, rank)` over one layer pays for the `d x d` covariances and
    eigendecompositions once instead of once per cell. At d=3584 that is the difference between
    minutes and tens of minutes per layer (E9's dose calibration sweeps 7 cells per layer).

    `need` selects which blocks to build:

    - `"leace"` -> `W_sqrt`, `W_inv` (from `_shrunk(cov(X), shrink)`) — the LEACE whitener.
    - `"wiener"` -> `S_inv_means` — the mean-span Wiener denominator `(FF^T + Sigma_c + Sigma_eps)^-1`.
    - `"wiener_marginal"` -> `Sigma_R`, `S_inv_marginal` — the `(Sigma_c + Sigma_R)^-1` denominator
      shared by the injected-`basis` and `sigma_g="instance"` Wiener branches.

    `mu`, `Delta`, `n_langs` and `shrink` are always present; `shrink` is checked on use, since a
    whitener built at one shrink is wrong for another.
    """
    X, lang_idx, problem_idx, langs = _pooled_np(acts_by_lang)
    L = len(langs)
    mu, Delta = _centered_mean_matrix(X, lang_idx)
    out: dict = {"mu": mu, "Delta": Delta, "n_langs": L, "shrink": float(shrink)}
    if "leace" in need:
        W_sqrt, W_inv = _sqrt_invsqrt(_shrunk(np.cov(X.T), shrink))
        out["W_sqrt"], out["W_inv"] = W_sqrt, W_inv
    if {"wiener", "wiener_marginal"} & set(need):
        pm = np.stack([X[problem_idx == p].mean(axis=0)
                       for p in range(int(problem_idx.max()) + 1)])
        Sigma_c = np.cov(pm.T)
        R = residualize_problem(X, problem_idx)
        if "wiener" in need:
            F = Delta / np.sqrt(L)
            Rl = R - np.stack([R[lang_idx == l].mean(axis=0) for l in range(L)])[lang_idx]
            out["S_inv_means"] = _inv_shrunk(F @ F.T + Sigma_c + np.cov(Rl.T), shrink)
        if "wiener_marginal" in need:
            Sigma_R = np.cov(R.T)
            out["Sigma_R"] = Sigma_R
            out["S_inv_marginal"] = _inv_shrunk(Sigma_c + Sigma_R, shrink)
    return out


def _check_whiten(whiten: dict, shrink: float, keys: tuple[str, ...]) -> None:
    """Fail loud on a whitener built for a different `shrink`, or missing the block this edit needs."""
    got = float(whiten.get("shrink", float("nan")))
    if not np.isclose(got, float(shrink)):
        raise ValueError(f"whitener was built at shrink={got}, this call asks for shrink={shrink}; "
                         f"rebuild it (the covariance ridge is baked into W_inv / S_inv)")
    missing = [k for k in keys if k not in whiten]
    if missing:
        raise ValueError(f"whitener lacks {missing}; rebuild it with layer_whitener(..., need=...) "
                         f"covering this edit (have {sorted(whiten)})")


def _centered_mean_matrix(X: np.ndarray, lang_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(mu, Delta)` — pooled mean `(d,)` and centered language means as columns `(d, L)`."""
    mu = X.mean(axis=0)
    L = int(lang_idx.max()) + 1
    m = np.stack([X[lang_idx == l].mean(axis=0) for l in range(L)])   # (L, d)
    return mu, (m - mu).T


def _t32(a: np.ndarray) -> Tensor:
    return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)


def _lang_stats(acts_by_lang: dict[str, Tensor], with_cov: bool = False) -> dict[str, dict]:
    """Per-language `{lang: {mu, sd[, Sigma]}}` in float64."""
    out = {}
    for lang, a in acts_by_lang.items():
        A = a.numpy().astype(np.float64)
        st = {"mu": A.mean(axis=0), "sd": A.std(axis=0)}
        if with_cov:
            st["Sigma"] = np.cov(A.T)
        out[lang] = st
    return out


def _target_stats(stats: dict[str, dict], target: str) -> dict:
    """Stats of the transport target: a language code, or `"barycenter"` = the average of the
    per-language moments (mean of means / stds / covariances — the cheap fixed-scale stand-in
    for the exact Bures barycenter, adequate for a lam-interpolated edit)."""
    if target != "barycenter":
        return stats[target]
    keys = [k for k in ("mu", "sd", "Sigma") if k in next(iter(stats.values()))]
    return {k: np.mean([s[k] for s in stats.values()], axis=0) for k in keys}


# --- erasure / estimation (language-agnostic) ----------------------------------------------------

def leace_edit(acts_by_lang: dict[str, Tensor], rank: int | None = None,
               shrink: float = 0.1, basis: Tensor | None = None,
               whiten: dict | None = None) -> AffineEdit:
    """LEACE-style oblique erasure of the language-mean span (arXiv 2306.03819).

    delta(h) = Sigma^{1/2} P Sigma^{-1/2} (h - mu), with P the orthogonal projector onto the
    *whitened* span of the centered language means Delta. At lam=1 all 11 post-edit language
    means coincide, so no linear probe can beat chance on language (perfect linear guardedness)
    — and among all affine maps achieving that, this one moves the activations least (minimal
    mean-squared edit). Contrast `ablate`: same span, but projected *orthogonally* in the raw
    Euclidean metric, which both over-edits high-variance nuisance coordinates (massive
    activations) and under-removes decodability per unit of edit energy (E6 Phase 0).

    Factored as `U = Sigma^{1/2} Q`, `V = Sigma^{-1/2} Q` with `Q` an orthonormal basis of the
    whitened span; rank defaults to L-1 (the mean span's full rank).

    `basis` (a `(d, r)` subspace, e.g. an angular `candidates.diffpca_subspace`) **injects that
    span in place of the language-mean span** — the SUBSPACE×OPERATOR cross cell "diffPCA span,
    erased obliquely in the whitened metric" (diffpca⊕oblique). rank then defaults to the basis's
    own column count. Default None => the standard LEACE mean span, unchanged.

    `whiten` (a `layer_whitener()` dict) supplies `mu`/`Delta`/`W_sqrt`/`W_inv` instead of
    recomputing them. None of those depend on `rank` or `basis`, so a caller sweeping either over
    one layer pays for the `d x d` covariance + eigendecomposition once. Byte-identical to the
    recompute path (`tests/test_operators.py`); `acts_by_lang` is then unused.
    """
    if whiten is not None:
        _check_whiten(whiten, shrink, ("mu", "Delta", "n_langs", "W_sqrt", "W_inv"))
        mu, Delta, n_langs = whiten["mu"], whiten["Delta"], whiten["n_langs"]
        W_sqrt, W_inv = whiten["W_sqrt"], whiten["W_inv"]
    else:
        X, lang_idx, _, langs = _pooled_np(acts_by_lang)
        mu, Delta = _centered_mean_matrix(X, lang_idx)
        Sigma = _shrunk(np.cov(X.T), shrink)
        W_sqrt, W_inv = _sqrt_invsqrt(Sigma)
        n_langs = len(langs)
    if basis is not None:
        span = np.asarray(basis, dtype=np.float64)           # injected diffPCA span (d, r)
        rank = rank if rank is not None else span.shape[1]
    else:
        span = Delta
        rank = rank if rank is not None else n_langs - 1
    Q, _ = np.linalg.qr(W_inv @ span)
    Q = Q[:, :rank]
    return AffineEdit(c=_t32(mu), U=_t32(W_sqrt @ Q), V=_t32(W_inv @ Q))


def wiener_edit(acts_by_lang: dict[str, Tensor], shrink: float = 0.1,
                sigma_g: str = "means", rank: int | None = None,
                basis: Tensor | None = None, whiten: dict | None = None) -> AffineEdit:
    """Random-effects Wiener shrinkage: remove the posterior-mean language effect.

    Model `h_i^l = mu + c_i + g_l + q_il + eps` (content / language / interaction+noise).
    The LMMSE estimate of the language component from one monolingual observation is

        E[g|h] = Sigma_g (Sigma_g + Sigma_c + Sigma_eps)^{-1} (h - mu),

    so `delta(h) = K (h - mu)` with the gain K estimated from the ANOVA scatters. `sigma_g`
    picks the language-covariance estimator — the TARGET knob of the E6 design matrix:

    - `"means"` (default): Sigma_g from the centered language means (`F = Delta / sqrt(L)`,
      `Sigma_g = F F^T`, rank <= L-1 exactly); denominator Sigma_g + Sigma_c + Sigma_eps
      (Sigma_c from per-problem means, Sigma_eps from doubly-centered residuals).
    - `"instance"`: Sigma_L = the top-`rank` (default L-1) eigencomponent of the residual-row
      covariance Sigma_R — language + interaction as the instances see it (the instance-target
      x soft-operator matrix cell); denominator Sigma_c + Sigma_R (the marginal of h - mu
      decomposes as Sigma_c + Sigma_R since residuals and problem means are uncorrelated).

    Either way K is rank <= L-1 / `rank`, stored factored. Unlike the hard projection (binary
    spectrum, one searched scalar lam), K attenuates each target direction by its
    signal-to-noise ratio snr/(1+snr) — the profile the E5 lam grid approximates with a single
    number. Most different from `ablate` where SNRs are low (7B trough: snr 20 -> 1.2).

    NOTE the `sigma_g="means"` branch **ignores `rank`**: `Sigma_g = FF^T` with `F = Delta/sqrt(L)`,
    and `Delta` is `(d, L)` by construction, so this edit is fixed at rank `L-1` and has no rank
    knob at all. Only the injected-`basis` and `"instance"` branches are rank-parameterized.
    `build_family_edits` refuses the `alg1 x wiener` rank cells for exactly this reason, rather
    than silently returning `rank` duplicates of the same edit.

    `whiten` (a `layer_whitener()` dict) supplies the rank/basis-independent second moments
    (`mu`, `Delta`, and whichever denominator this branch needs) instead of recomputing them.
    Byte-identical to the recompute path; `acts_by_lang` is then unused.
    """
    if whiten is not None:
        _need = (("S_inv_means",) if (basis is None and sigma_g == "means")
                 else ("Sigma_R", "S_inv_marginal"))
        _check_whiten(whiten, shrink, ("mu", "Delta", "n_langs") + _need)
        mu, Delta, L = whiten["mu"], whiten["Delta"], whiten["n_langs"]
        Sigma_c = R = lang_idx = None
        return _wiener_from_moments(mu, Delta, L, Sigma_c, R, lang_idx, shrink,
                                    sigma_g, rank, basis, whiten)
    X, lang_idx, problem_idx, langs = _pooled_np(acts_by_lang)
    L = len(langs)
    mu, Delta = _centered_mean_matrix(X, lang_idx)
    pm = np.stack([X[problem_idx == p].mean(axis=0)
                   for p in range(int(problem_idx.max()) + 1)])       # (n, d) problem means
    Sigma_c = np.cov(pm.T)
    R = residualize_problem(X, problem_idx)                           # content removed
    return _wiener_from_moments(mu, Delta, L, Sigma_c, R, lang_idx, shrink,
                                sigma_g, rank, basis, None)


def _wiener_from_moments(mu, Delta, L, Sigma_c, R, lang_idx, shrink,
                         sigma_g, rank, basis, whiten) -> AffineEdit:
    """The Wiener gain from either freshly computed or cached (`whiten`) second moments.

    Split out so `whiten=` and the recompute path share ONE definition of the three branches —
    the numerics below are what `wiener_edit` documents, and there is no second copy to drift.
    """
    if basis is not None:
        # diffpca⊕wiener cross cell: soft (snr/(1+snr)) shrinkage of an INJECTED diffPCA span.
        # Sigma_g = F F^T with F = each diffPCA direction scaled by its residual-covariance energy.
        B = np.asarray(basis, dtype=np.float64)                       # (d, r) diffPCA span
        Sigma_R = whiten["Sigma_R"] if whiten else np.cov(R.T)        # language + interaction + noise
        e = np.einsum("dj,de,ej->j", B, Sigma_R, B)                  # m_j^T Sigma_R m_j per direction
        F = B * np.sqrt(np.maximum(e, 0.0))
        S_inv = (whiten["S_inv_marginal"] if whiten
                 else _inv_shrunk(Sigma_c + Sigma_R, shrink))
    elif sigma_g == "means":
        F = Delta / np.sqrt(L)                            # Sigma_g = F F^T; rank L-1, NOT `rank`
        if whiten:
            S_inv = whiten["S_inv_means"]
        else:
            Rl = R - np.stack([R[lang_idx == l].mean(axis=0) for l in range(L)])[lang_idx]
            Sigma_eps = np.cov(Rl.T)                                  # interaction + noise
            S_inv = _inv_shrunk(F @ F.T + Sigma_c + Sigma_eps, shrink)
    elif sigma_g == "instance":
        rank = rank if rank is not None else L - 1
        Sigma_R = whiten["Sigma_R"] if whiten else np.cov(R.T)        # language + interaction + noise
        evals, evecs = np.linalg.eigh(Sigma_R)
        top = np.argsort(evals)[::-1][:rank]
        F = evecs[:, top] * np.sqrt(np.maximum(evals[top], 0.0))      # Sigma_L = F F^T
        S_inv = (whiten["S_inv_marginal"] if whiten
                 else _inv_shrunk(Sigma_c + Sigma_R, shrink))
    else:
        raise ValueError(f"unknown sigma_g: {sigma_g!r} (want 'means' or 'instance')")
    return AffineEdit(c=_t32(mu), U=_t32(F), V=_t32(S_inv @ F))


def wdiffpca_edit(acts_by_lang: dict[str, Tensor], rank: int | None = None,
                  shrink: float = 0.1) -> AffineEdit:
    """Whitened-diffPCA oblique erasure — the (instance-target x oblique-operator) matrix cell.

    diffPCA's span, but estimated in the whitened metric (PCA of `R @ Sigma^{-1/2}`, which
    neutralizes the massive-coordinate leverage the angular hack only half-fixes) and removed
    obliquely like `leace_edit` (`U = Sigma^{1/2} B`, `V = Sigma^{-1/2} B`). Screen-only
    condition: it completes the target x operator design matrix so a diffPCA loss cannot be
    blamed on the raw metric or the orthogonal operator. In the scratch pre-validation it
    still lost to the mean-span cells at both 1.5B and 7B.
    """
    X, _, problem_idx, langs = _pooled_np(acts_by_lang)
    rank = rank if rank is not None else len(langs) - 1
    mu = X.mean(axis=0)
    Sigma = _shrunk(np.cov(X.T), shrink)
    W_sqrt, W_inv = _sqrt_invsqrt(Sigma)
    R = residualize_problem(X, problem_idx)
    _, _, Vt = np.linalg.svd(R @ W_inv, full_matrices=False)          # PCA in whitened coords
    B = Vt[:rank].T
    return AffineEdit(c=_t32(mu), U=_t32(W_sqrt @ B), V=_t32(W_inv @ B))


def edit_energy(edit: AffineEdit, X: Tensor) -> float:
    """Unit-lam edit energy fraction `E||delta(x)||^2 / E||x||^2` over the rows of `X`.

    The dose unit for the accuracy-vs-edit-energy frontier (E6 "dose comparability"): a
    schedule's total dose is `sum_l lam_l^2 * e_l` with `e_l` this fraction at layer `l`
    (exact for the linear part; the bias term contributes at lam^2 as well). Operators differ
    by 10-40x in `e_l` at equal lam, so raw lam values are NOT comparable across arms — this
    is the common scale.
    """
    X = X.float()
    d = edit.delta(X)
    return float((d ** 2).sum() / (X ** 2).sum())


def uniform_shrink_edit(acts_by_lang: dict[str, Tensor], eps: float | None = None,
                        rank: int | None = None) -> AffineEdit:
    """Energy-matched nonspecific control: `h <- (1 - lam*eps) h`.

    A direction-free shrink whose edit energy matches what alg1's hard projection removes on
    the same data (`eps = sqrt(E||h @ M_s||^2 / E||h||^2)`, uncentered — the projection acts on
    raw h). Distinguishes "removing *language*" from "removing *that much energy*", the confound
    the rank-matched random subspace cannot address (it removes ~r/d, far less). Pass `eps`
    explicitly to bypass the matching.
    """
    if eps is None:
        X, _, _, langs = _pooled_np(acts_by_lang)
        rank = rank if rank is not None else len(langs) - 1
        W = language_means(acts_by_lang)
        M = build_language_subspace(W, rank)[1].numpy().astype(np.float64)
        eps = float(np.sqrt(((X @ M) ** 2).sum() / (X ** 2).sum()))
    d = next(iter(acts_by_lang.values())).shape[1]
    return AffineEdit(diag=torch.full((d,), float(eps), dtype=torch.float32))


def projection_edit(M_s: Tensor) -> AffineEdit:
    """Wrap a `(d, r)` orthonormal subspace as an `AffineEdit` (`delta = M_s M_s^T h`).

    Exactly `intervene.ablate` — lets alg1/diffpca run through `build_affine_hooks` so every E6
    condition shares one code path.
    """
    return AffineEdit(U=M_s.float(), V=M_s.float())


# --- transports (language-conditional: one edit per source language) -----------------------------

def meanshift_edit(acts_by_lang: dict[str, Tensor], source: str,
                   target: str = "barycenter") -> AffineEdit:
    """Rung 1 — pure mean transport: `h <- h - lam*(mu_s - mu_t)` (diff-in-means)."""
    stats = _lang_stats(acts_by_lang)
    t = _target_stats(stats, target)
    return AffineEdit(b=_t32(stats[source]["mu"] - t["mu"]))


def adain_edit(acts_by_lang: dict[str, Tensor], source: str, target: str = "barycenter",
               eps: float = 1e-8) -> AffineEdit:
    """Rung 2 — AdaIN diagonal moment matching (Huang & Belongie 2017 applied per-coordinate):
    at lam=1, `h <- mu_t + (sd_t/sd_s) * (h - mu_s)`."""
    stats = _lang_stats(acts_by_lang)
    s, t = stats[source], _target_stats(stats, target)
    scale = t["sd"] / np.maximum(s["sd"], eps)
    return AffineEdit(c=_t32(s["mu"]), diag=_t32(1.0 - scale), b=_t32(s["mu"] - t["mu"]))


def gaussian_ot_edit(acts_by_lang: dict[str, Tensor], source: str, target: str = "barycenter",
                     kind: str = "bures", shrink: float = 0.1,
                     rank: int | None = None) -> AffineEdit:
    """Rung 3 — full-covariance Gaussian transport (CORAL 1511.05547 / Bures OT, cf. AcT
    2410.23054, Representation Surgery 2402.09631).

    At lam=1, `h <- mu_t + A (h - mu_s)` with
    - `kind="bures"`: the optimal-transport map between the Gaussian fits,
      `A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}`;
    - `kind="coral"`: the CORAL recoloring `A = Sigma_t^{1/2} Sigma_s^{-1/2}` (no rotation
      correction; kept as the classic ladder rung).

    `I - A` is dense; `rank=None` stores it as-is (d^2 floats — build band layers only),
    `rank=r` keeps its top-r SVD factors.
    """
    stats = _lang_stats(acts_by_lang, with_cov=True)
    s, t = stats[source], _target_stats(stats, target)
    Ss, St = _shrunk(s["Sigma"], shrink), _shrunk(t["Sigma"], shrink)
    if kind == "coral":
        St_sqrt, _ = _sqrt_invsqrt(St)
        _, Ss_inv = _sqrt_invsqrt(Ss)
        A = St_sqrt @ Ss_inv
    elif kind == "bures":
        Ss_sqrt, Ss_inv = _sqrt_invsqrt(Ss)
        M_sqrt, _ = _sqrt_invsqrt(Ss_sqrt @ St @ Ss_sqrt)
        A = Ss_inv @ M_sqrt @ Ss_inv
    else:
        raise ValueError(f"unknown gaussian_ot kind: {kind!r}")
    M = np.eye(A.shape[0]) - A                                       # delta = M z + (mu_s - mu_t)
    b = _t32(s["mu"] - t["mu"])
    c = _t32(s["mu"])
    if rank is None:
        return AffineEdit(c=c, dense=_t32(M), b=b)
    P, sv, Qt = np.linalg.svd(M)
    return AffineEdit(c=c, U=_t32(P[:, :rank] * sv[:rank]), V=_t32(Qt[:rank].T), b=b)


def translator_edit(acts_by_lang: dict[str, Tensor], source: str, target: str = "en",
                    ridge: float = 0.1, rank: int | None = None) -> AffineEdit:
    """Rung 4 — paired ridge translator fit on problem-matched pairs (the uniquely-parallel rung).

    Regress the paired counterfactual residual `y_i = (h_i^t - mu_t) - (h_i^s - mu_s)` on
    `z_i = h_i^s - mu_s` (same problem i in both languages — supervision none of the unpaired
    transports see), giving `C` with `T(h) = mu_t + z + z C`. At lam the edit interpolates
    `h <- h + lam*(T(h) - h)`; unlike the static rungs the correction `z C` is input-dependent.

    Solved in the n-dimensional data subspace via SVD of Z (n << d), ridge
    `gamma = ridge * mean(S^2)`; `rank` optionally truncates the correction. Stored factored
    (`V = Q P_g`, `U = -(Q_g S_g)`), never d x d.
    """
    Xs = acts_by_lang[source].numpy().astype(np.float64)
    Xt = acts_by_lang[target].numpy().astype(np.float64)
    assert Xs.shape == Xt.shape, "translator needs index-aligned parallel activations"
    mu_s, mu_t = Xs.mean(axis=0), Xt.mean(axis=0)
    Z, Y = Xs - mu_s, (Xt - mu_t) - (Xs - mu_s)
    P, S, Qt = np.linalg.svd(Z, full_matrices=False)                 # Z = P S Q^T, k = min(n, d)
    gamma = ridge * float((S ** 2).mean())
    G = (S / (S ** 2 + gamma))[:, None] * (P.T @ Y)                  # (k, d): C = Q G
    if rank is None:
        return AffineEdit(c=_t32(mu_s), U=_t32(-G.T), V=_t32(Qt.T), b=_t32(mu_s - mu_t))
    Pg, Sg, Qgt = np.linalg.svd(G, full_matrices=False)
    return AffineEdit(c=_t32(mu_s), U=_t32(-(Qgt[:rank].T * Sg[:rank])),
                      V=_t32(Qt.T @ Pg[:, :rank]), b=_t32(mu_s - mu_t))


# --- per-layer builder (the build_layer_subspaces twin) -------------------------------------------

_BUILDERS = {
    "leace": leace_edit,
    "wiener": wiener_edit,
    "wdiffpca": wdiffpca_edit,
    "shrink": uniform_shrink_edit,
    "meanshift": meanshift_edit,
    "adain": adain_edit,
    "gaussian_ot": gaussian_ot_edit,
    "translator": translator_edit,
}


def build_layer_edits(acts: dict[str, Tensor], layers, *, name: str,
                      source: str | None = None, basis_by_layer: dict | None = None,
                      whitener_by_layer: dict | None = None,
                      **kw) -> dict[int, AffineEdit]:
    """Per-layer edits `{layer -> AffineEdit}` for one estimator `name` — the affine twin of
    `candidates.build_layer_subspaces`, consuming the same `{lang: (n, n_layers, d)}` cache.

    Language-agnostic names (`EDIT_NAMES`) ignore `source`; transports (`TRANSPORT_NAMES`)
    require it (the eval language whose runs the hooks will edit) and forward `target`/kind
    kwargs. `basis_by_layer` (a `{layer -> (d, r)}` subspace map, e.g. a per-layer angular diffPCA
    basis from `candidates.build_layer_subspaces`) injects that span into `leace`/`wiener` per
    layer — the SUBSPACE×OPERATOR cross (diffpca⊕oblique / diffpca⊕wiener). `whitener_by_layer`
    (a `{layer -> layer_whitener(...)}` map) is forwarded as `whiten=` the same way, so a caller
    sweeping rank or basis over fixed layers reuses each layer's covariance work. Slot the result
    into `intervene.AffineSpec(edits=..., lam=schedule)`.
    """
    if name not in _BUILDERS:
        raise ValueError(f"unknown edit {name!r}; know {sorted(_BUILDERS)}")
    if name in TRANSPORT_NAMES:
        if source is None:
            raise ValueError(f"{name!r} is language-conditional: pass source=<eval lang>")
        kw = {"source": source, **kw}
    build = _BUILDERS[name]
    out: dict[int, AffineEdit] = {}
    for layer in layers:
        sl = {l: a[:, layer, :] for l, a in acts.items()}
        lkw = dict(kw)
        if basis_by_layer is not None:
            lkw["basis"] = basis_by_layer[int(layer)]      # inject this layer's diffPCA span
        if whitener_by_layer is not None:
            lkw["whiten"] = whitener_by_layer[int(layer)]  # reuse this layer's second moments
        out[int(layer)] = build(sl, **lkw)
    return out


# --- E9: the (basis x operator) family join -------------------------------------------------------
#
# The trade-off grid (docs/workshop/tradeoff_grid.md) sweeps six families = {alg1, diffpca} x
# {orthogonal projection, LEACE, Wiener}. The two halves already exist -- `build_layer_subspaces`
# estimates the span, `_BUILDERS` fits the operator -- but nothing joins them, because
# `projection_edit` takes a prebuilt `(d, r)` subspace rather than raw activations and so cannot
# live in `_BUILDERS`. `e6_phase2_behavioral.py:245` papers over the seam with an `isinstance`
# test; this is that seam, named and validated once.

#: subspace estimators the grid crosses
FAMILY_BASES: tuple[str, ...] = ("alg1", "diffpca")
#: operators the grid crosses ("proj" == the hard orthogonal projection of `intervene.ablate`)
FAMILY_OPERATORS: tuple[str, ...] = ("proj", "leace", "wiener")


def family_cell_absent(basis: str, operator: str, rank: int, n_langs: int) -> str | None:
    """Why the `(basis, operator, rank)` cell does not exist, or None if it does.

    Two cells are **algebraically** absent, not merely expensive, and both must be refused rather
    than silently answered with a duplicate of a neighbouring cell:

    1. `alg1` above rank `L-1`. Alg1's span is built from the L language means, so it has at most
       `L-1` dimensions; `r=32` exists for diffPCA only.
    2. `alg1 x wiener` at any rank other than `L-1`. The mean-span Wiener gain is
       `Sigma_g = FF^T` with `F = Delta/sqrt(L)`; `Delta` is `(d, L)` by construction, so
       `wiener_edit(sigma_g="means")` never reads its `rank` argument and the edit is pinned at
       rank `L-1`. Asking for `r=1` there returns the `r=L-1` edit — which on a rank sweep plots
       as a flat plateau that is really one point measured repeatedly.
    """
    if basis not in FAMILY_BASES:
        return f"unknown basis {basis!r}; know {list(FAMILY_BASES)}"
    if operator not in FAMILY_OPERATORS:
        return f"unknown operator {operator!r}; know {list(FAMILY_OPERATORS)}"
    if rank < 1:
        return f"rank must be >= 1, got {rank}"
    cap = n_langs - 1
    if basis == "alg1" and rank > cap:
        return (f"alg1 spans at most L-1 = {cap} dimensions (built from {n_langs} language means), "
                f"so rank {rank} does not exist; r>{cap} is a diffpca-only cell")
    if basis == "alg1" and operator == "wiener" and rank != cap:
        return (f"alg1 x wiener has no rank knob: the mean-span gain is pinned at rank L-1 = {cap} "
                f"(F = Delta/sqrt(L) is (d, L) by construction), so rank {rank} would silently "
                f"return the rank-{cap} edit")
    return None


def build_family_edits(acts: dict[str, Tensor], layers, *, basis: str, operator: str, rank: int,
                       shrink: float = 0.1, whitener_by_layer: dict | None = None,
                       span_by_layer: dict | None = None,
                       diffpca_angular: bool = True) -> dict[int, AffineEdit]:
    """Per-layer `AffineEdit`s for one `(basis, operator, rank)` cell of the E9 trade-off grid.

    Puts all six families on the single `AffineSpec` / `build_affine_hooks` path:

    - `proj` -> `projection_edit(build_layer_subspaces(name=basis, rank=rank)[layer])`, i.e.
      exactly `intervene.ablate` wrapped as an affine edit.
    - `leace` / `wiener` with `basis="alg1"` -> the operator's own native language-mean span.
    - `leace` / `wiener` with `basis="diffpca"` -> the angular diffPCA span injected per layer
      (`basis_by_layer`), the SUBSPACE x OPERATOR cross cell.

    Raises `ValueError` on the two algebraically absent cells (see `family_cell_absent`) instead of
    returning a duplicate.

    Two caches, for a caller sweeping many cells over fixed layers (E9's calibrator):
    `whitener_by_layer` reuses each layer's covariance work, and `span_by_layer` supplies the
    `{layer -> (d, rank)}` subspace this cell would otherwise rebuild. `span_by_layer` is used by
    `proj` (as the projector) and by `diffpca x leace/wiener` (as the injected basis); the
    `alg1 x leace/wiener` cells fit their own native language-mean span and ignore it. Its column
    count is checked against `rank`, so a mis-keyed cache fails loud instead of silently running
    the wrong rank.
    """
    layers = [int(l) for l in layers]
    why = family_cell_absent(basis, operator, rank, len(acts))
    if why is not None:
        raise ValueError(f"no such family cell {basis}x{operator} r={rank}: {why}")

    def _span() -> dict:
        if span_by_layer is None:
            return build_layer_subspaces(acts, layers, name=basis, rank=rank,
                                         diffpca_angular=diffpca_angular)
        out = {}
        for l in layers:
            M = span_by_layer[l]
            if int(M.shape[1]) != int(rank):
                raise ValueError(f"span_by_layer[{l}] has rank {M.shape[1]}, cell asks for {rank}")
            out[l] = M
        return out

    if operator == "proj":
        return {l: projection_edit(M) for l, M in _span().items()}

    kw: dict = {"shrink": shrink}
    if basis == "diffpca":
        # inject the diffPCA span; leace takes its rank from the basis's own column count, and
        # wiener's basis branch scales those columns by their residual-covariance energy.
        kw["basis_by_layer"] = _span()
    elif operator == "leace":
        kw["rank"] = rank                     # truncate the QR of the whitened mean span
    # alg1 x wiener: rank is pinned at L-1 and refused above, so nothing to pass.
    return build_layer_edits(acts, layers, name=operator,
                             whitener_by_layer=whitener_by_layer, **kw)
