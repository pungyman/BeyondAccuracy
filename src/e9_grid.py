"""The E9 trade-off-grid arm set — ONE definition, shared by the calibrator and the sweep.

`docs/workshop/tradeoff_grid.md` specifies a cross, not a factorial: the centre is Zhao's schedule
(rank L-1, their bands, their published global lambda), and each axis is varied with the other held
at centre. Six families = {alg1, diffpca} x {orthogonal projection, LEACE, Wiener}.

This lives in `src/` rather than in either script because `e9_dose_calibrate.py` and
`e9_tradeoff_grid.py` must enumerate the *same* cells: the calibrator freezes a spec and a lambda
per cell, the sweep looks them up by name. A drift between two local copies of the cell list would
be silent — the sweep would skip arms, or ask for lambdas that were never solved.

Naming: `e9_<basis>_<operator>_r<rank>_d<rung x 100>`, e.g. `e9_alg1_proj_r10_d100` (the centre) or
`e9_diffpca_wiener_r32_d150`. Filename-safe, sortable, and parseable back to the four fields — the
figure code groups by (basis, operator) to get its six families. The arm lands in the per-item CSV's
`cond` column, matching every existing consumer (`results/E8/items/<model>__<shard>.csv`'s filename
suffix is a *language shard*, not an arm).

Two cells the spec asks for do not exist; see `operators.family_cell_absent`:

- `alg1` above rank L-1 — its span is built from the L language means.
- `alg1 x wiener` off rank L-1 — `wiener_edit(sigma_g="means")` never reads `rank`, so those cells
  would be duplicate generations of that family's centre plotted as a rank plateau.

So the grid is **37 arms + clean**, not the 39 the doc first estimated.
"""
from __future__ import annotations

from dataclasses import dataclass

from operators import FAMILY_BASES, FAMILY_OPERATORS, family_cell_absent

#: dose rungs as multiples of the cross-language anchor dose; `CENTRE_RUNG` is Zhao's lambda verbatim
RUNGS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
CENTRE_RUNG: float = 1.0
#: off-centre rank cells per basis. The centre rank itself is L-1, resolved from the cache.
RANK_AXIS: dict[str, tuple[int, ...]] = {"alg1": (1, 4), "diffpca": (1, 4, 32)}
#: stage order (tradeoff_grid.md): every stage is a complete figure on its own
STAGES: tuple[str, ...] = ("clean", "centre", "lambda", "rank")

CLEAN = "clean"


@dataclass(frozen=True)
class Arm:
    """One grid cell. `stage` is which sweep stage populates it; `rung` is a dose multiple."""
    basis: str
    operator: str
    rank: int
    rung: float
    stage: str

    @property
    def name(self) -> str:
        return arm_name(self.basis, self.operator, self.rank, self.rung)

    @property
    def family(self) -> str:
        """The figure's grouping key — one of the six (basis x operator) families."""
        return f"{self.basis}_{self.operator}"

    @property
    def cell(self) -> tuple[str, str, int]:
        """The `(basis, operator, rank)` key a frozen edit-set is stored under (rung-independent)."""
        return (self.basis, self.operator, self.rank)


def arm_name(basis: str, operator: str, rank: int, rung: float) -> str:
    """`e9_<basis>_<operator>_r<rank>_d<rung x 100>` — the `cond` string stamped on every row."""
    return f"e9_{basis}_{operator}_r{int(rank)}_d{int(round(float(rung) * 100)):03d}"


def parse_arm(cond: str) -> dict | None:
    """Inverse of `arm_name`, or None for anything that is not an E9 arm (e.g. `clean`, A1 arms).

    Returning None rather than raising is deliberate: the figure and metric code read pooled CSVs
    that mix E9 arms with `clean` and the A1 `<arm>_tuned` cells.
    """
    if not cond.startswith("e9_"):
        return None
    parts = cond[3:].rsplit("_", 2)
    if len(parts) != 3 or not parts[1].startswith("r") or not parts[2].startswith("d"):
        return None
    head, r, d = parts
    bits = head.split("_", 1)
    if len(bits) != 2:
        return None
    basis, operator = bits
    try:
        rank, rung = int(r[1:]), int(d[1:]) / 100.0
    except ValueError:
        return None
    if basis not in FAMILY_BASES or operator not in FAMILY_OPERATORS:
        return None
    return {"basis": basis, "operator": operator, "rank": rank, "rung": rung,
            "family": f"{basis}_{operator}"}


def centre_rank(n_langs: int) -> int:
    """Zhao's rank: L-1, the structural cap of the language-mean span (`Config.rank`)."""
    return n_langs - 1


def grid_arms(n_langs: int, *, rungs=RUNGS, rank_axis=None) -> list[Arm]:
    """The grid's cells in stage order: 6 centre, then the lambda axis, then the rank axis.

    Stage order matters — `tradeoff_grid.md` wants every stage to be a complete figure across all
    languages, so a walltime kill between stages still leaves something publishable. Cells that
    `operators.family_cell_absent` rejects are skipped here, so the sweep never asks for them.
    """
    rank_axis = RANK_AXIS if rank_axis is None else rank_axis
    r0 = centre_rank(n_langs)
    arms: list[Arm] = []

    def add(basis, operator, rank, rung, stage):
        if family_cell_absent(basis, operator, rank, n_langs) is None:
            arms.append(Arm(basis, operator, rank, rung, stage))

    for basis in FAMILY_BASES:                       # stage 1 -- the six centre points
        for operator in FAMILY_OPERATORS:
            add(basis, operator, r0, CENTRE_RUNG, "centre")
    for basis in FAMILY_BASES:                       # stage 2 -- the dose axis at centre rank
        for operator in FAMILY_OPERATORS:
            for rung in rungs:
                if rung != CENTRE_RUNG:
                    add(basis, operator, r0, rung, "lambda")
    for basis in FAMILY_BASES:                       # stage 3 -- the rank axis at centre dose
        for operator in FAMILY_OPERATORS:
            for rank in rank_axis.get(basis, ()):
                add(basis, operator, rank, CENTRE_RUNG, "rank")
    return arms


def grid_cells(arms) -> list[tuple[str, str, int]]:
    """The distinct `(basis, operator, rank)` edit-sets the arms need, deduped, in first-seen order.

    Edits are rung-independent (a rung only rescales lambda), so the calibrator builds and freezes
    one edit-set per cell and the sweep reuses it across that cell's rungs.
    """
    seen: dict[tuple[str, str, int], None] = {}
    for a in arms:
        seen.setdefault(a.cell, None)
    return list(seen)
