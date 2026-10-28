"""The matplotlib settings every paper figure must share.

Two rules, and both are compliance rather than taste:

  * **`pdf.fonttype: 42`.** matplotlib's default is 3, which emits Type 3 fonts. They are embedded,
    so they pass the letter of the ACL requirement that every font be embedded, but they render
    badly on screen, do not subset cleanly, and some venue checkers flag them. 42 gives TrueType.
    `ps.fonttype` matters only if a figure is ever routed through EPS; set it for the same reason.

  * **A serif family matching the document.** ACL: "fonts in figures should conform to the document
    fonts." The paper is Times via `\\usepackage{times}`, so figures are Times too. `make_figs_e3rank`
    and `make_figs_e10` previously set no family at all and came out in DejaVu Sans, which is why
    the E3 and E10 figures did not look like they belonged to the paper.

`assert_serif_available()` exists because matplotlib falls back **silently** when a named font is
missing -- you get DejaVu and a warning buried in the log, and the figure ships wrong. Call it once
at the top of a figure script so a missing Times fails loudly on a machine that lacks it.

Import before creating any figure:

    import paperstyle; paperstyle.apply()

Per-script tuning (sizes, colours) goes in the script, after `apply()`.
"""
from __future__ import annotations

import matplotlib as mpl

SERIF = ["Times New Roman", "STIX Two Text", "DejaVu Serif"]

BASE = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "stix",   # so $\rho$ in an axis label is serif too, not DejaVu
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 200,
}


def apply(**overrides) -> None:
    mpl.rcParams.update(BASE)
    if overrides:
        mpl.rcParams.update(overrides)


def assert_serif_available() -> None:
    """Fail loudly rather than silently shipping the fallback font."""
    from matplotlib import font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    if not any(name in have for name in SERIF):
        raise SystemExit(
            f"none of {SERIF} is installed, so matplotlib would silently fall back to a sans "
            f"default and the figures would not match the paper's Times. Install one, or extend "
            f"paperstyle.SERIF with a serif face this machine has."
        )
