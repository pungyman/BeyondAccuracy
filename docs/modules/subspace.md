# `subspace`

Source: [src/subspace.py](../../src/subspace.py)

Builds the **language subspace** `M_s` from per-language mean activations — the core of
Algorithm 1. This is the heart of the package; the derivation is in [theory.md](../theory.md).

All functions compute on `LINALG_DEVICE` (CPU) in fp32, because `torch.linalg.svd` on MPS is
unreliable. See [config](config.md).

## Functions

### `language_means(acts) -> Tensor`

Stack per-language mean activations into the mean matrix `W = [m_1 | ... | m_L]`.

- **`acts`** — `{lang: (n_samples, d)}` (a single layer's activations from
  [extract](extract.md)).
- **Returns** — `W` of shape **`(d, L)`**, fp32/CPU. Column `i` is the mean activation of the
  `i`-th language (insertion order of the dict, i.e. `cfg.langs`).

### `build_language_subspace(W, rank) -> tuple[Tensor, Tensor]`

The **two-step SVD** of Algorithm 1. Mirrors the numpy reference `build_aligner` exactly.

- **`W`** — `(d, L)` mean matrix.
- **`rank`** — subspace dimension (default `cfg.rank = L - 1`).
- **Returns** — `(wc_new, M_s)`:
  - `wc_new` — the refined anchor (shared-centroid direction), shape `(d,)`.
  - `M_s` — orthonormal **language subspace basis**, shape **`(d, rank)`**.

Steps (see [theory.md](../theory.md) for the math):

1. Anchor `wc = W @ 1 / L`; center `W - wc 1^T`.
2. First SVD -> `Ws = U[:, :rank]`, `Gamma = Vh.T[:, :rank] * S[:rank]`.
3. Best-fit reconstruction `best_fit = wc 1^T + Ws Gamma^T`.
4. Refine anchor via pseudo-inverse: `wc_new = pinv(best_fit).T @ 1`, then normalize by
   `(wc_new**2).sum()`.
5. Deflate `best_fit - wc_new 1^T`, second SVD -> `M_s = U2[:, :rank]`.

```python
from subspace import language_means, build_language_subspace, projector
W = language_means(acts_by_lang)              # (d, L)
wc, M_s = build_language_subspace(W, rank=10) # M_s: (d, 10)
```

### `projector(M_s) -> Tensor`

Orthogonal projector onto `span(M_s)`: `P = M_s @ M_s.T`, shape `(d, d)`. Sign/rotation
invariant, so it is the right object for comparing two subspaces independent of basis choice.

### `build_aligner(rank, lan_emb)`

**Reference numpy implementation**, copied verbatim from
`mlrs/src/main_multilingual_space.py`. It exists as ground truth: `build_language_subspace`
(torch) must match it. Not used in the hot path — prefer the torch version for real work.

- **`lan_emb`** — `{lang: array (n_samples, d)}`.
- **Returns** — `(wc_new, ws_new)`, the numpy analogues of `(wc_new, M_s)`.

### `build_subspace_variant(W, rank, mode="uniform", heavy_index=0, heavy_weight=3.0)`

E1 anchor experiments — different ways of choosing what to project out before extracting the
language subspace. Returns `(M_s, meta)` where `meta` includes `effective_rank` and `mode`.

| `mode` | What it does | `effective_rank` |
|--------|--------------|------------------|
| `"uniform"` | Standard two-step builder with the uniform anchor `1` (equivalent to `build_language_subspace`). | `rank` |
| `"weighted"` | Anchor weight vector with `heavy_weight` on `heavy_index` — concentrates the anchor on a single (non-English) language. | `rank` (also reports `heavy_index`) |
| `"rank3"` | Subtract a **rank-3 shared** (language-agnostic) subspace — the top-3 left singular vectors of the **uncentered** `W` (dominated by the common centroid) — then SVD the residual. `M_s` is the residual language variation. | `rank - 2` (also reports `shared_rank=3`) |
| `"none"` | Plain centered SVD with no anchor refinement (`U[:, :rank]` of `W - wc 1^T`). | `rank` |

- Unknown `mode` raises `ValueError`.
- The `"weighted"` mode relies on a private helper `_anchored_subspace(W, rank, weight)` — the
  generalized two-step builder where the anchor is read off by an arbitrary `weight` vector
  instead of `1`.

```python
from subspace import build_subspace_variant
M_s, meta = build_subspace_variant(W, rank=10, mode="rank3")
meta   # {"effective_rank": 8, "mode": "rank3", "shared_rank": 3}
```

## Related

- [theory.md](../theory.md) — full derivation of the two-step SVD and the projector.
- [intervene](intervene.md) — uses `M_s` to ablate / swap.
- [controls](controls.md) — null subspaces compared against `M_s`.
- [extract](extract.md) — produces `acts_by_lang` for `language_means`.
