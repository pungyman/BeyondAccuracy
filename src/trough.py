"""Where the rho-vs-layer curve bottoms out — with an interior guard.

Deliberately dependency-free (no torch, no config, no pandas). Three consumers import it and one
of them, `paper/workshop/figures/make_figs_e3rank.py`, runs under a plotting interpreter that has
matplotlib but no torch, so this cannot live in `geometry.py`.

## Why the guard exists

"The trough" had always been a plain global argmin over layers. That is correct for every model
measured up to 2026-08-31 and silently wrong for the first one that has an early-layer dip.
On the post-`enable_thinking` Qwen3-8B cache the rank-10 `alg1` minimum is **layer 1, at 3% depth**
— an embedding-adjacent artifact, not the mid-band language subspace the campaign is about, which
sits at **L21 (60%)**. Measured troughs across the three models we have are 59-67% relative depth;
nothing near an edge has ever been the real answer.

That single argmin fed four places, so one model with an early dip produced three wrong outputs at
once (the sweep's printed read-out and its rank table, the figure's trough marker, and the
figure's whole rank-curve panel) and stood ready to seed E10's intervention band at layer 1.

## Why it reports rather than silently masks

`global_layer` comes back alongside, so callers can say when the two disagree. A guard that
quietly substituted a different answer would be the same class of bug one level up: if some future
model genuinely troughs near an edge, that is a finding, and it should be visible rather than
clipped away. `describe` formats the one-line disclosure.

The window is generous on purpose — [0.15, 0.95] of depth excludes the embedding-adjacent layers
and the final one without coming close to the 59-67% band where the real troughs live.
"""
from __future__ import annotations

LO, HI = 0.15, 0.95


def interior_trough(layers, values, *, lo: float = LO, hi: float = HI):
    """Argmin of `values`, restricted to layers in the interior `[lo, hi]` of the stack.

    `layers` and `values` are equal-length sequences; layers need not be sorted or contiguous.
    Returns `(layer, value, global_layer, global_value)` — the guarded answer first, the
    unguarded one alongside so the caller can disclose a disagreement.

    Falls back to the global argmin when the window would be empty (a curve with fewer than a
    handful of layers), because an empty window is a worse answer than an unguarded one.
    """
    pairs = list(zip(layers, values))
    if not pairs:
        raise ValueError("interior_trough: no layers")
    g_layer, g_value = min(pairs, key=lambda lv: lv[1])
    span = max(layers)
    if span <= 0:
        return g_layer, g_value, g_layer, g_value
    inner = [(l, v) for l, v in pairs if lo <= l / span <= hi]
    if not inner:
        return g_layer, g_value, g_layer, g_value
    layer, value = min(inner, key=lambda lv: lv[1])
    return layer, value, g_layer, g_value


def depth(layer, layers) -> float:
    """Relative depth of `layer` in `[0, 1]`, using the deepest layer present as the denominator."""
    span = max(layers)
    return (layer / span) if span > 0 else 0.0


def describe(layer, value, g_layer, g_value, layers) -> str:
    """One line naming the trough, and the override when the global argmin disagrees."""
    line = f"rho={value:.4f} @ layer {layer} ({depth(layer, layers):.0%} depth)"
    if g_layer != layer:
        line += (f"  [interior guard: the global argmin is layer {g_layer} "
                 f"({depth(g_layer, layers):.0%} depth, rho={g_value:.4f}) — reported as an "
                 f"edge artifact, see src/trough.py]")
    return line
