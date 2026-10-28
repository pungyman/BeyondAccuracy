# Theory

This page explains the math behind this project: what the **language subspace** is, how
Algorithm 1 builds it from activations, and what the probe / ablation / swap / control
experiments measure. Implementation lives in
[subspace.py](../src/subspace.py), [intervene.py](../src/intervene.py),
and [controls.py](../src/controls.py).

## Background

Zhao et al. observe that, in a multilingual LLM, the residual stream encodes two things that
can be partly disentangled: **what** is being reasoned about (content) and **which language**
it is being expressed in (language identity). They hypothesize that language identity lives
in a low-dimensional **language subspace** of the residual stream, and that projecting it out
("ablation") or replacing it ("swap") changes the surface language while largely preserving
the reasoning.

This project reimplements that machinery against a parallel multilingual MATH corpus: the
same problems translated into \(L = 11\) languages (`bn, de, en, es, fr, ja, ru, sw, te, th, zh`),
so any activation difference at a fixed problem index is attributable to language, not content.

## The mean matrix \(W\)

For each language \(\ell\) we collect the model's final-token residual-stream vector at a
chosen layer for \(n\) parallel prompts, then average to a per-language **mean activation**
\(m_\ell \in \mathbb{R}^d\) (\(d = d_{\text{model}}\)). Stacking these as columns gives

\[
W = [\,m_1 \mid m_2 \mid \dots \mid m_L\,] \in \mathbb{R}^{d \times L}.
\]

This is exactly [`language_means`](modules/subspace.md): mean over the sample axis per
language, stacked column-wise. Because content is held fixed across the parallel prompts, the
spread of the columns of \(W\) is dominated by language.

## Algorithm 1 — the two-step SVD

The goal is an orthonormal basis \(M_s \in \mathbb{R}^{d \times r}\) (with `rank` \(r = L - 1\)
by default) whose span is the language subspace. A naive top-\(r\) SVD of the centered \(W\)
mixes the **shared centroid** (a content/anchor direction common to all languages) into the
basis. Algorithm 1 removes that anchor in a second pass. Implemented as
[`build_language_subspace`](modules/subspace.md):

1. **Anchor estimate.** \(w_c = \tfrac{1}{L} W \mathbf{1}\) — the mean of the language means
   (the common centroid).
2. **First SVD.** Center and decompose: \(W - w_c \mathbf{1}^\top = U S V^\top\). Take
   \(W_s = U_{:, :r}\) and \(\Gamma = V_{:, :r} S_{:r}\).
3. **Best-fit reconstruction.** \(\widehat{W} = w_c \mathbf{1}^\top + W_s \Gamma^\top\) —
   the rank-(\(r{+}1\)) reconstruction of the language means.
4. **Refine the anchor.** Recompute the anchor that the reconstruction implies via a
   pseudo-inverse: \(w_c' = (\widehat{W}^{+})^\top \mathbf{1}\), normalized so
   \(\lVert w_c' \rVert^2 = 1\)-ish (`wc_new /= (wc_new**2).sum()`).
5. **Deflate and re-SVD.** \(\widehat{W} - w_c' \mathbf{1}^\top = U_2 S_2 V_2^\top\); the
   **language subspace** is \(M_s = U_{2,\,:, :r}\).

The two-step refinement matters: it pins down the anchor \(w_c'\) self-consistently so the
returned \(M_s\) captures *between-language variation* rather than the dominant shared
centroid. The torch implementation is checked against a verbatim numpy reference
[`build_aligner`](modules/subspace.md) (ported from
`mlrs/src/main_multilingual_space.py`) to guard against drift.

> All of this runs on CPU in fp32 (`LINALG_DEVICE`) because `torch.linalg.svd` on MPS is
> unreliable. See [config](modules/config.md).

## Algorithm 1 is PCA, twice over

It is tempting to read Algorithm 1 as a bespoke "two-step SVD" with no relation to the
familiar baselines. It is not. **Algorithm 1 is principal component analysis, run twice — and
the *only* thing that distinguishes it from "PCA on the language means" is the choice of
center.** This section makes that precise, then shows that the whole five-subspace
ladder (research-workspace reference) collapses to a *single*
operation indexed by two knobs. Once you see it, the experimental design (E3/E4) reads
differently.

### PCA *is* the SVD of a centered matrix

Recall the textbook identity. Given points \(\{x_1,\dots,x_N\}\subset\mathbb R^d\) stacked as
the columns of \(Y\in\mathbb R^{d\times N}\), PCA subtracts the centroid \(\bar y =
\tfrac1N Y\mathbf 1\) and takes the SVD of the centered matrix:

\[
Y - \bar y\,\mathbf 1^\top = U S V^\top,\qquad
(Y-\bar y\mathbf 1^\top)(Y-\bar y\mathbf 1^\top)^\top
= \sum_i (x_i-\bar y)(x_i-\bar y)^\top = N\,\mathrm{Cov}(\{x_i\}).
\]

The columns of \(U\) — the left singular vectors of the centered matrix — are the principal
directions (eigenvectors of the covariance). The load-bearing word is **centered**: the
directions PCA returns depend entirely on *what you subtract first*. Subtract a different
reference point and you diagonalize a different second-moment matrix, so you get a different
subspace. Hold onto that — it is the entire plot.

> **Orientation aside.** When samples are *columns* (our \(W\in\mathbb R^{d\times L}\)) the
> principal directions are the **left** singular vectors \(U\). When samples are *rows* (the
> pooled \(X\in\mathbb R^{(nL)\times d}\) the probe/control tooling uses) they are the
> **right** singular vectors \(V\). Same PCA, transposed bookkeeping.

### Step 1 (the first SVD) is *exactly* PCA on the eleven language means

Algorithm 1's first decomposition centers \(W\) by the centroid \(w_c=\tfrac1L W\mathbf 1\)
and takes the SVD of \(W - w_c\mathbf 1^\top\). By the identity above, its left singular
vectors are the principal axes of the eleven-point cloud \(\{m_1,\dots,m_L\}\):

\[
(W-w_c\mathbf 1^\top)(W-w_c\mathbf 1^\top)^\top
= \sum_\ell (m_\ell-w_c)(m_\ell-w_c)^\top = L\,\mathrm{Cov}(\{m_\ell\}).
\]

That is *verbatim* the `none` variant
([`build_subspace_variant(W, r, mode="none")`](../src/subspace.py)). So step 1 is not a warm-up
for something more exotic — **step 1 is "PCA on the language means," full stop** (this is the
equivalence in [shared_methods §4.2](experiments/shared_methods.md#42-none-centered-means-pca--pca-on-the-means)).

### Step 2 (the second SVD) is PCA *again* — about a learned anchor, not the centroid

The second decomposition takes the SVD of \(\widehat W - w_c'\mathbf 1^\top\) and returns
\(M_s = U_{2,\,:,:r}\). This is once more the SVD of a *centered* matrix — hence PCA — but it
is centered at the **learned anchor \(w_c'\)** rather than at the arithmetic centroid of
\(\widehat W\)'s columns. Concretely, \(M_s\) diagonalizes the second moment measured *from
\(w_c'\)*:

\[
(\widehat W - w_c'\mathbf 1^\top)(\widehat W - w_c'\mathbf 1^\top)^\top
= \sum_\ell (\hat m_\ell - w_c')(\hat m_\ell - w_c')^\top,
\]

and the top-\(r\) left singular vectors maximize \(\sum_\ell\lVert U^\top(\hat m_\ell -
w_c')\rVert^2\) — variance *about the anchor*, not about the mean. **The single way Algorithm 1
departs from textbook PCA is this choice of origin**, and it is a deliberate modeling decision,
not an accident of implementation.

What *is* \(w_c'\)? It is the dual direction read off every reconstructed language column via
the pseudo-inverse, \(w_c' = (\widehat W^{+})^\top\mathbf 1\) (then normalized): the direction
"equally present" across all \(L\) languages — the **shared/common component**. Deflating it
leaves the residual span as pure between-language variation. Critically, \(w_c'\) is in general
**not** the centroid \(w_c\), and need not even be an affine combination of the \(m_\ell\) —
which is *exactly* why the mean-difference residual is nonzero
([shared_methods §4.1](experiments/shared_methods.md#41-mean-difference-residuals-are-near-tautological-circularity))
and why Algorithm 1 differs from `none` at all.

> **The punchline so far.**
> `none` = PCA on the means, centered at the **centroid** \(w_c\).
> **Alg1** = PCA on the (rank-\(r\) cleaned) means, centered at the **learned anchor** \(w_c'\).
> The entire difference between the incumbent subspace and the naive baseline is *the center*.

### The unifying picture: every candidate is one PCA, indexed by (population × centering)

Now generalize. The five subspaces (research-workspace reference)
under comparison are **not** five algorithms. Four of them are the *same* operation — take the
top-\(r\) principal directions of a centered activation matrix — and they differ along exactly
**two axes**:

1. **Population** — *which cloud of points* you decompose: the \(L\) language **means**
   (columns of \(W\)), or all \(nL\) **instances** (rows of \(X\)).
2. **Centering** — *what you subtract* before the SVD: the **uniform centroid**, a **learned
   anchor** \(w_c'\), or the **per-problem content mean**.

Lay the four PCA-based subspaces on that 2-D grid:

| PCA computed over … | centered at **uniform centroid** | centered at **learned anchor** \(w_c'\) | centered **per problem** (content removed) |
|---|---|---|---|
| **\(L\) language means** (\(W\), \(d\times L\)) | `none` | **Alg1** (+ rank-\(r\) cleanup) | — |
| **\(nL\) instances** (\(X\), \(nL\times d\)) | **pooled-PCA** | — | **DiffPCA** |

`random` is the one genuine outsider: it is PCA of *nothing* — a Haar-random rank-\(r\)
subspace — present only as the dimensionality floor.

Two rows of this grid say something you might otherwise miss for months:

- **pooled-PCA and DiffPCA are the same PCA over the same population** (all \(nL\) instances).
  The *only* difference is the center: pooled-PCA subtracts the **global** mean
  (`Xc = X - X.mean(0)` in [`pooled_pca_subspace`](../src/controls.py)), DiffPCA subtracts the
  **per-problem** mean (`residualize_problem` in [probe.py](../src/probe.py)). Subtract the
  wrong center and the very same PCA returns generic *content* directions; subtract the
  per-problem content and it returns *language* directions. This is *why* pooled-PCA is the
  correct "variance ≠ language" null — it differs from the E4 proposal by a centering choice
  and nothing else.
- **Alg1 and `none` are the same PCA over the same population** (the \(L\) means), differing
  only by center (anchor vs centroid). This is *why* `none` is the rung that isolates the value
  of Algorithm 1's anchor refinement
  ([baseline ladder](experiments/shared_methods.md#6-baseline-ladder)).

### What this does to the E3/E4 question

Read through the grid, the headline question of E3/E4 — *is the reconstruction-optimal subspace
(DiffPCA) also the intervention-optimal one (Alg1)?* — is a question about **which of the two
PCA knobs matters for clean intervention**. DiffPCA-vs-Alg1 turns *both* knobs at once
(means → instances, **and** anchor → per-problem content). The `none` and pooled-PCA rungs are
precisely the "turn one knob" conditions that let you attribute any DiffPCA win to the
*population* swap versus the *centering* swap. So the baseline ladder is not a courtesy set of
controls — it is the **factorial design** that decomposes the headline contrast into its two
PCA ingredients.

> **Precise caveat (so we don't oversell the word "PCA").** Step 2 is PCA only *up to the
> center*: it diagonalizes the second moment about \(w_c'\), not the covariance about
> \(\mathrm{mean}(\widehat W)\). Re-centering at \(\widehat W\)'s own column centroid would
> yield yet another rank-\(r\) subspace — neither Alg1 nor `none`. The anchor is a modeling
> choice with consequences, which is the whole reason this section exists. SVD and PCA are
> "deeply connected" precisely because PCA *is* an SVD of centered data; the art is entirely in
> deciding what counts as the center.

## The projector

Once you have \(M_s\), the orthogonal projector onto the language subspace is

\[
P = M_s M_s^\top \in \mathbb{R}^{d \times d},
\]

implemented as [`projector`](modules/subspace.md). \(P\) is invariant to the choice of basis
(any rotation of \(M_s\) within its span gives the same \(P\)), which makes it the right
object for comparing subspaces.

## Interventions

Given an activation \(h \in \mathbb{R}^{d}\) (or a batch), the two core edits live in
[intervene.py](../src/intervene.py):

- **Ablation** removes the language component:
  \[
  \text{ablate}(h) = h - (h M_s) M_s^\top = (I - P)\,h.
  \]
- **Swap** (paper Eq. 3') strips the source-language coordinates, then recenters from the
  source mean \(\mu_s\) to the target mean \(\mu_t\):
  \[
  \text{swap}(h) = \text{ablate}(h) - \mu_s + \alpha\,\mu_t.
  \]
  \(\alpha\) scales the injected target-language mean (1.0 = full swap).

These are applied in-flight via residual-stream forward hooks on the decoder blocks
([`interventions`](modules/intervene.md)). Which token positions they affect is set by
`InterventionSpec.scope`, and the default — used for every reported number — is `"prompt_only"`:
the prefill pass only. See below.

### Scope of the intervention: which tokens and which layers

A natural worry: we *estimate* `M_s` from the **final token**
([`extract_last_token_per_layer`](modules/extract.md)), so do we also only *ablate* the final
token? **No — estimation and application are separate.** The hook rewrites the block's entire
output tensor `output[0]` of shape `(batch, seq_len, d)`, so the projection hits **every prompt
token position**, not just the last one ([intervene.py](../src/intervene.py)).

This matters because of the residual + attention dataflow. If you ablated only the last
position, every *earlier* token would keep its language component in the KV cache, and the next
layer's attention would read it straight back into the last token — the ablation would leak.
Ablating all positions closes that path. (The *direction* is still estimated from the last
token because that position has attended over the whole prompt and is the most
language-saturated; only the *application* is broadcast to all positions.)

**Generated tokens are NOT edited under the default scope.** `InterventionSpec.scope` defaults to
`"prompt_only"`, which is what every reported E5–E8 number uses, and it means *prefill only*:

- **Generative path.** `_layer_fn` returns `h` unchanged when `scope == "prompt_only" and
  h.shape[-2] == 1` — a sequence length of 1 is exactly a decode step, so the edit applies during
  the prompt forward and is skipped for every generated token.
- **Likelihood path.** There is no decode loop, so the scope becomes a position mask over the
  single `[context + candidate]` forward: `edit_mask` marks the `ctx_len` context positions and
  leaves the scored continuation untouched.
- `"all_steps"` is the alternative — every decode step / all positions — and is **currently
  unused** by any reported result.

So the ablation reshapes how the *question* is represented and then lets the model generate
freely; it is not a continuous steer during generation. This is a deliberate choice to match
Zhao et al.'s Appendix F, and it is one of the open questions in `open.md` (research-workspace reference) (Q2),
because their released code does not do what their appendix says.

**What Zhao et al. actually do** (confirmed from their released code,
[`MuyuenLP/Language-Reasoning-Disentangle`](https://github.com/MuyuenLP/Language-Reasoning-Disentangle)).
Two scoping choices, neither of which is "last token only":

1. **Positions — all prompt tokens, and (contrary to their paper) generated ones too.** The
   ablation (`ForSteeringVLLMMlrs.def_hook_steer_fn`) operates on the full `output[0]` =
   `(batch, seq_len, d)` — i.e. **all prompt tokens**, never a single last-token slice
   (last-token slicing appears only in their *extraction* hook, `output[0][0][-1]`). Their runner
   also defaults to `steering_level="prompt"`, and the only other mode (`"all"`) is commented out.
   Appendix F states it plainly: *"we ablate activations only on the tokens in the prompt."*

   **But the released code does not honour that.** `steering_level` selects only *which hook path
   is installed*; `ForSteeringVLLM.generate_token` then wraps the entire vLLM
   `self.model.generate(...)` call in `add_hooks`, so the hook also fires on every decode step
   (where `output[0]` is `(batch, 1, d)`, the single new token). Their effective scope is therefore
   **prefill + decode**, i.e. our `"all_steps"`, while their paper describes our `"prompt_only"`.
   Verified by reading the repo verbatim (2026-08-13). We follow their *paper*; the discrepancy and
   the planned A/B are tracked as Q2 in `open.md` (research-workspace reference), and it is one of the remaining
   candidate explanations for the response-fidelity gap between our replication and their Table 1.
2. **Layers — ablate the middle, *re-inject* the top.** They use a **per-layer** subspace
   (`vector.pt`, indexed by layer) and **two layer bands with opposite-signed strengths**:
   positive at middle layers (remove language) and **negative** at higher layers (add it back).
   Their `scripts/r1-7b_mgsm.sh` for DeepSeek-R1-Distill-Qwen-7B (28 layers) is explicit: ablate
   layers **10–19** with strength \(\in\{0, .1, .2, .3\}\), re-inject layers **20–27** with
   strength \(\in\{0, -.1, -.2, -.3\}\) (the edit is \(h - \text{strength}\cdot P h\), so a
   negative strength *amplifies* the language component). This operationalizes the §4.2 finding:
   removing language mid-network helps reasoning, but the **top layers must keep language** or
   the model cannot produce fluent target-language output (language fidelity collapses to
   English).

**How this repo differs** (so comparisons stay honest). Two paths must be distinguished here.
The bare `ablate`/`swap` primitives, registered directly as hooks via `interventions(model, hooks)`
— the E1–E4 route — carry **no scope logic**, so they use a **single** `M_s` at the chosen layer(s)
applied uniformly to **all positions on the prompt *and* every generation step**, with **no
re-injection band**. That is *more* aggressive than the paper on two axes: it edits the generated
tokens too, and it lacks the top-layer re-injection that preserves fluency. The **E5+ reported
path** (`InterventionSpec` -> `build_layer_hooks`) is scope-aware and defaults to `"prompt_only"`,
so it edits the prompt only and *does* use the two-band schedule — that is the path behind every
number in E5–E8, and the paragraph above applies to it only in the `"all_steps"` condition. The `interventions` hook *can* express the
paper's scheme — `hooks` is a per-layer `{layer_idx: fn}` map, so you can register `ablate`
(positive) at mid layers and a negative-coefficient re-injection at high layers — but you must
construct it. Treat **(a) prompt-only vs all-steps** and **(b) uniform vs
ablate-mid/re-inject-high** as explicit intervention knobs in
E4 Phase 3 (research-workspace reference); the
paper-faithful setting is the right baseline, the repo's blanket ablation a legitimate
stronger-ablation variant.

### Prompt-only vs all-steps: what the scope actually means

Generation has two phases: a **prefill** pass (the whole prompt goes through at once, filling the
KV cache) and then **decode** (one token per step, each attending back over the cache). The
ablation hook rewrites a block's output `(batch, seq, d)`, so *when* it fires decides its reach:

- **Prompt-only** — the edit applies during **prefill only**. It scrubs language identity from the
  model's *understanding of the question*; the edited prompt representations are written into the
  KV cache, so every generated token attends over a "de-accented" prompt, **but the generated
  tokens themselves are never edited** — the output stays fluent in the target language. Intuition:
  *understand the question with its linguistic accent stripped, then answer in your natural voice.*
  This is why the paper pairs prompt-only ablation with high-layer re-injection: reason
  language-invariantly, but keep output fidelity.
- **All-steps** — the edit **also fires on every decode step**, projecting language out of each
  token as it is emitted. The per-step edits **accumulate** over the sequence and destabilize the
  output language, pulling it toward the model's dominant language (the Chinese-attractor collapse
  E4 Phase 3 logged, ~47% Han at \(\lambda=1\)). Intuition: *keep suppressing your accent while you
  also speak → garbled output.* Legitimate as a *stronger-ablation* variant, but not fidelity-
  preserving.

These are **two orthogonal axes**: positions/steps (prompt-only vs all-steps) and layers
(mid-ablate / high-reinject). For **log-likelihood-scored** tasks (no decode loop — the multiple-
choice/cloze benchmarks in E5) the scope becomes a **position mask** over a single
`[context + candidate]` forward pass: prompt-only edits only the context positions (scoring the
candidate under the edited context); all edits the candidate positions too. E5 executes and reports
**prompt-only** and treats all-steps as a labeled follow-up
(E5 §4 (research-workspace reference),
[benchmarks.md §5](modules/benchmarks.md#5-modular-intervention-machinery)).

## Controls — what would falsify "M_s encodes language"

A subspace that changes behavior is only meaningful relative to a null. [controls.py](../src/controls.py)
provides three baselines (see [controls](modules/controls.md)):

- **Haar-random subspace** — a generic rank-\(r\) direction set with no language structure.
  Tests whether the *dimensionality* alone explains an effect.
- **Pooled-PCA subspace** — top-\(r\) high-variance directions of the language-collapsed
  activations: generic content directions, language-agnostic. Tests whether *high variance*
  alone explains an effect.
- **Permuted-label subspace** — shuffle which language each sample belongs to, then rebuild
  via the same two-step SVD. Tests whether the construction would "find" a subspace even when
  language labels are meaningless.

If the real \(M_s\) drives effects (probe accuracy, language flips) far beyond all three
nulls, the language-subspace hypothesis survives.

## How the probe fits in

The probe ([probe.py](../src/probe.py)) measures **how decodable language is** from
activations — before and after an intervention. The subtlety is leakage: parallel
translations of the same problem share content, so a naive split lets the probe memorize
problems. Hence the leakage-free tools — group cross-validation by problem
([`leave_problems_out_probe`](modules/probe.md)), content residualization
([`residualize_problem`](modules/probe.md)), and an empirical chance floor from label
permutation ([`shuffled_label_floor`](modules/probe.md)). See [probe](modules/probe.md).
