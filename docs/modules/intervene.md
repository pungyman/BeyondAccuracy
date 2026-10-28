# `intervene`

Source: [src/intervene.py](../../src/intervene.py)

Edits the residual stream to remove or replace the language subspace, applied in-flight via
forward hooks. The math (ablation, swap / Eq. 3') is derived in [theory.md](../theory.md).

All functions move `M_s` / means onto the activation's device and dtype, so you can keep the
fp32/CPU `M_s` from [subspace](subspace.md) and let the hook cast it to match the model.

## Functions

### `ablate(h, M_s) -> Tensor`

Remove the language-subspace component of `h`:

```
ablate(h) = h - (h @ M_s) @ M_s.T = (I - P) h
```

- **`h`** — activations, `(..., d)`.
- **`M_s`** — subspace basis `(d, rank)`.
- **Returns** — `h` with its projection onto `span(M_s)` removed.

### `swap(h, M_s, mu_s, mu_t, alpha=1.0) -> Tensor`

Paper **Eq. 3'**: strip the source-language subspace coordinates, then recenter from the
source mean to the target mean.

```
swap(h) = ablate(h, M_s) - mu_s + alpha * mu_t
```

- **`mu_s`, `mu_t`** — source / target per-language mean activations, `(d,)`. Compute as
  `acts_by_lang[lang].mean(0)`.
- **`alpha`** — scale on the injected target mean (`1.0` = full swap).
- **Returns** — edited activations whose surface language is pushed toward the target.

### `interventions(model, hooks)`

A context manager that registers residual-stream **forward hooks** on decoder blocks and
removes them cleanly on exit (even on error).

- **`hooks`** — `{layer_idx: fn}` where `fn(h) -> h` edits the block output. The hook handles
  the common case where a block returns a tuple `(hidden, ...)` by editing only `output[0]`
  and passing the rest through unchanged.
- Registered on `model.model.layers[idx]`. **Fires on every forward pass** — the prompt pass
  *and* each generation step — so the intervention persists throughout `generate`.

```python
from intervene import swap, interventions

mu = {lang: a.mean(0) for lang, a in acts_by_lang.items()}
hooks = {layer: lambda h: swap(h, M_s, mu["ja"], mu["en"], alpha=1.0)}

with interventions(model, hooks):
    out = model.generate(**enc, max_new_tokens=128)
# hooks are removed here
```

> Because the hook fires on every step, the edit applies to the freshly generated tokens too,
> not just the prompt — which is what makes the language flip persist through decoding.

## Related

- [subspace](subspace.md) — provides `M_s`.
- [scoring](scoring.md) — measure whether a swap actually changed the output language and kept
  the answer correct.
- [theory.md](../theory.md) — ablation / swap derivation.
