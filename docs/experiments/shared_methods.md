# Shared methods, metrics, and protocol

Common definitions for E3 (research-workspace reference) and
E4 (research-workspace reference). Notation follows the glossary (research-workspace reference) and
[theory.md](../theory.md).

## 1. Data and activations

We use the parallel multilingual MATH corpus: the same \(n\) problems translated into
\(L = 11\) languages, index-aligned, so any activation difference at a fixed problem index is
attributable to **language**, not content ([corpus.py](../../src/corpus.py)).

Activations are the **final-token residual stream at a fixed layer**, extracted with
[`extract_last_token_per_layer`](../modules/extract.md) → shape `(n, n_layers, d)` per
language. For a chosen layer we slice to `acts_by_lang[ℓ]` of shape `(n, d)`. All linear
algebra runs on CPU/fp32 (`LINALG_DEVICE`).

We organize the per-instance activations two equivalent ways:

- **Per language**: `acts_by_lang = {ℓ: (n, d)}`.
- **Pooled long form**: a stacked matrix \(X \in \mathbb{R}^{(nL)\times d}\) with aligned
  arrays `lang_idx` (which language) and `problem_idx` (which problem) — the form the probe
  tooling expects (`leave_problems_out_probe`, `residualize_problem`).

> **Sample-size note.** `Config.n_samples = 32` is fine for iterating, but DiffPCA estimates a
> 10-dim subspace and benefits from more instances. Use **\(n \approx 100\text{–}200\)** for
> the estimation runs; keep 32 for quick checks. Always report \(n\).

## 2. Steering / translation vectors

For an ordered language pair \((s, t)\):

- **Per-prompt** (instance-level): \(v_i^{s\to t} = h_i^t - h_i^s\), one per problem \(i\).
- **Mean** (the swap direction): \(\bar v^{\,s\to t} = m_t - m_s\), the average of the above.

There are \(\binom{L}{2} = 55\) unordered and \(L(L-1) = 110\) ordered pairs.

**Order and the residual.** The captured-fraction metric depends only on \(\lVert P v\rVert /
\lVert v\rVert\), and \(P(-v)\) has the same norm as \(Pv\). So for *residual magnitude*, the
110 ordered pairs collapse to **55 distinct values** — direction is irrelevant. Keep the full
110 only for *directed* analyses (e.g. comparing \(\bar v^{s\to t}\) to the swap's
\(\mu_s\!\to\!\mu_t\), or signed per-axis coordinates).

## 3. The five subspaces

See the table in the README (research-workspace reference). All are rank
\(r = 10\), built at the same layer from the same activations. The three "real" candidates
(Alg1, none, DiffPCA) are language-derived; pooled-PCA and random are nulls.

## 4. Three equivalences you must keep in mind

These prevent us from claiming trivialities as discoveries and pin down what is actually novel.

### 4.1 Mean-difference residuals are near-tautological (circularity)

The mean steering vector \(\bar v^{s\to t} = m_t - m_s\) is a difference of columns of \(W\),
hence lies in the column space of the centered matrix \(W - w_c\mathbf 1^\top\) (rank \(\le
L-1 = 10\)). If \(M_s\) were exactly that column space, \(\rho(\bar v) \equiv 1\) by
construction. Algorithm 1's residual is nonzero **only** because the two-step refinement uses
a different anchor (\(w_c' \neq w_c\), and \(w_c'\) is generally not an affine combination of
the \(m_\ell\)). So:

> The **mean-difference** residual is a *consistency check* with a known near-zero expectation
> (it quantifies the anchor refinement), **not** a discovery. The **per-prompt** residual is
> the real signal.

### 4.2 `none` (centered-means PCA) = "PCA on the means"

Because centered \(W\) has rank \(\le 10 = r\), its top-\(r\) left singular vectors span the
*entire* mean-difference space. That is exactly
`build_subspace_variant(W, 10, mode="none")`. So the "PCA of the language means" idea is
already implemented as the `none` variant, and `theory.md` argues Algorithm 1 improves on it
by removing the shared centroid. **`none` is therefore a baseline, not the E4 proposal.**

### 4.3 DiffPCA ≡ PCA on problem-residualized activations (the E4 object)

Maximizing the per-prompt residual captured over all pairs,
\(\max_{P:\,\mathrm{rank}=r}\ \sum_{i}\sum_{s,t}\lVert P\,v_i^{s\to t}\rVert^2\),
reduces to PCA on the within-problem-centered residuals \(r_i^\ell = h_i^\ell -
\tfrac1L\sum_k h_i^k\). Sketch:

\[
\sum_{s,t}\lVert P(r^t - r^s)\rVert^2
= 2L\sum_\ell \lVert P r^\ell\rVert^2 - 2\Big\lVert P\!\sum_\ell r^\ell\Big\rVert^2
= 2L\sum_\ell \lVert P r^\ell\rVert^2,
\]

since \(\sum_\ell r_i^\ell = 0\). Summing over \(i\), the objective is
\(2L\sum_{i,\ell}\lVert P r_i^\ell\rVert^2\), whose rank-\(r\) maximizer is the top-\(r\) PCA
of the residual matrix. And `residualize_problem(X, problem_idx)` subtracts exactly the
within-problem mean \(\tfrac1L\sum_k h_i^k\) (each problem has \(L\) rows, one per language).
Therefore:

> **DiffPCA `M_s` = top-\(r\) right singular vectors of `residualize_problem(X)`.**

A content-controlled, instance-rich language subspace, almost free to build. This is the
*only* genuinely new construction in E3/E4.

```python
from sklearn.decomposition import PCA
from probe import residualize_problem
# X: (n*L, d) pooled activations at the layer; problem_idx: (n*L,)
R = residualize_problem(X, problem_idx)        # within-problem content removed
M_s_diff = PCA(n_components=10).fit(R).components_.T   # (d, 10)
```

### 4.4 Synthesis: the candidates are one PCA, indexed by (population × centering)

The three equivalences above are facets of a single fact, derived in full in
[theory.md → *Algorithm 1 is PCA, twice over*](../theory.md#algorithm-1-is-pca-twice-over):
**every language-derived subspace here is the same operation** — top-\(r\) principal directions
of a *centered* activation matrix (PCA = SVD of centered data) — differing along only two axes:

1. **Population** — the \(L\) language **means** (columns of \(W\)) or all \(nL\) **instances**
   (rows of \(X\)).
2. **Centering** — the **uniform centroid**, the **learned anchor** \(w_c'\), or the
   **per-problem content mean**.

| PCA over … | centered at **uniform centroid** | centered at **learned anchor** \(w_c'\) | centered **per problem** |
|---|---|---|---|
| **\(L\) means** (\(W\)) | `none` | **Alg1** (+ rank-\(r\) cleanup) | — |
| **\(nL\) instances** (\(X\)) | **pooled-PCA** | — | **DiffPCA** |

(`random` is the only non-PCA member — the dimensionality floor.) Two consequences pin down the
ladder below: **pooled-PCA and DiffPCA are the same PCA over the same instances**, differing
*only* in centering (global mean vs per-problem mean) — which is why pooled-PCA is the right
"variance ≠ language" null; and **Alg1 and `none` are the same PCA over the same means**,
differing *only* in centering (anchor vs centroid) — which is why `none` isolates the anchor
refinement. The DiffPCA-vs-Alg1 contrast therefore turns *both* knobs at once, and the baseline
ladder is the factorial design that separates the population swap from the centering swap.

## 5. Metrics

### 5.1 Geometry of a single vector
- **Captured fraction**: \(\rho(v) = \lVert P v\rVert^2 / \lVert v\rVert^2 \in [0,1]\).
- **Residual energy**: \(1 - \rho(v) = \lVert (I-P)v\rVert^2 / \lVert v\rVert^2\).

### 5.2 Aggregate residual (per pair / per language / per layer)
- Mean \(\rho\) over instances; report a CI by bootstrapping over problems.
- Per-language summary: average \(\rho\) over all pairs involving \(\ell\).
- Per-layer summary: average \(\rho\) over all pairs/instances at that layer.

### 5.3 Subspace-vs-subspace comparison (no labels, no generation)
- **Principal angles** \(\theta_1\le\dots\le\theta_r\) between two subspaces
  (`torch.linalg.svdvals(M_a.T @ M_b)` → \(\cos\theta\)).
- **Projector distance**: \(\lVert P_a - P_b\rVert_F = \sqrt{2}\,\lVert\sin\Theta\rVert_2\)
  — basis-invariant; the right "how different are these subspaces" number.
- **Subspace overlap / captured energy**: \(\mathrm{tr}(P_a P_b)/r\) — fraction of \(b\)'s
  basis captured by \(a\).

### 5.4 Disentanglement (probe-based, used in E4)
- **Language decodability**: accuracy of `leave_problems_out_probe` (group-CV by problem) on
  raw vs. ablated activations. Lower-after-ablation = more language removed.
- **Chance floor**: `shuffled_label_floor` (permuted labels). The target after ablation.
- **Reasoning/content retention**: a content probe that should be *preserved* after ablating
  the language subspace — e.g. predict the answer bucket, problem index family, or operation
  type. Disentanglement = language → floor while content probe holds.

### 5.5 Behavior (generation-based, used in E4)
- **Language flip rate**: fraction of generations whose output language (`scoring.detlang` /
  `script_dom` / `coherent_english`) becomes the target under swap, or loses the source under
  ablation.
- **Reasoning preservation**: MATH accuracy `scoring.is_correct(extract_pred(out), gold)`
  under intervention vs. clean, per language.

## 6. Baseline ladder

Every claim ("`M_s` captures the translation", "DiffPCA disentangles better") is reported
**relative to** this ladder, never in isolation:

1. **random** rank-10 — dimensionality floor. \(\mathbb{E}[\rho] \approx r/d = 10/1536 \approx
   0.65\%\) for an arbitrary vector. Anything language-meaningful sits far above this.
2. **pooled-PCA** rank-10 — high-variance *content* directions, language-agnostic. Tests
   whether sheer variance explains an effect. Concretely, for a rank-10 subspace it runs PCA on
   the \(nL\) pooled activations and takes the ten largest variance directions; no problem
   alignment, language labels, or language-specific nuance enter the construction.
3. **none** rank-10 — PCA-on-means; isolates the value of Algorithm 1's anchor refinement and
   of DiffPCA's instance-level estimation.
4. **Alg1** — the incumbent `M_s`.
5. **DiffPCA** — the E4 proposal.

(The first three map to `controls.random_orthonormal_subspace`,
`controls.pooled_pca_subspace`, and `subspace.build_subspace_variant(mode="none")`.)

## 7. Statistical protocol

- **Train/test split on problems.** Fit every subspace (`M_s`, DiffPCA, controls) and every
  probe on a *train* set of problem indices; evaluate residuals/probes/behavior on a *held-out*
  set. DiffPCA has many more effective parameters than Alg1 — held-out evaluation guards
  against overfitting and makes Alg1-vs-DiffPCA a fair fight. Split by `problem_idx` (never let
  a problem's translations straddle train/test). The intended generalization test is: derive
  the language-specific subspace \(M_s\) from train problems only, freeze it, then intervene
  with that same \(M_s\) on unseen test problems and measure whether performance improves.
- **Permutation nulls.** Use `shuffled_label_floor` and `controls.permuted_label_subspace` to
  establish chance for probe and subspace-construction claims respectively.
- **Paired comparisons.** Alg1 vs DiffPCA are evaluated on the *same* prompts/instances;
  compare with paired tests (e.g. paired bootstrap or Wilcoxon over problems) and report
  effect sizes, not just p-values.
- **Seeds.** Fix `Config.seed`; average controls (random subspace) over ≥5 seeds.
- **Layer convention.** Report the full layer sweep; call out the middle layer
  (`n_layers // 2`, the default `intervene_layer`) since that is where interventions act.

## 8. Code surface reused

| Need | Where |
|------|-------|
| parallel prompts | `corpus.load_parallel` |
| activations | `extract.extract_last_token_per_layer` |
| `W`, Alg1 `M_s` | `subspace.language_means`, `subspace.build_language_subspace` |
| projector | `subspace.projector` |
| `none` baseline | `subspace.build_subspace_variant(mode="none")` |
| DiffPCA | `probe.residualize_problem` + `sklearn.decomposition.PCA` |
| nulls | `controls.random_orthonormal_subspace`, `controls.pooled_pca_subspace`, `controls.permuted_label_subspace` |
| probes | `probe.leave_problems_out_probe`, `probe.shuffled_label_floor` |
| interventions | `intervene.ablate`, `intervene.swap`, `intervene.interventions` |
| scoring | `scoring.is_correct`, `scoring.extract_pred`, `scoring.detlang`, `scoring.coherent_english` |
