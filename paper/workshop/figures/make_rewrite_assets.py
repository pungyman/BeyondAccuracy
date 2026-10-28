#!/usr/bin/env python3
"""Generate the figures and compact tables used by the 2026-09-07 rewrite.

All inputs are committed numerical summaries or CSVs produced by the analysis
pipeline.  Run from anywhere; outputs are written beside this script.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]
sys.path.insert(0, str(HERE))
import paperstyle  # noqa: E402

paperstyle.assert_serif_available()
paperstyle.apply(
    **{
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.25,
    }
)

ALG = "#0072B2"
DIFF = "#D55E00"
GRID_COLORS = {"alg1": ALG, "diffpca": DIFF}
GRID_MARKERS = {"proj": "o", "leace": "s", "wiener": "^"}

PROBE_COLORS = {
    ("frozen", "alg1"): "#D55E00",
    ("frozen", "diffpca"): "#E69F00",
    ("retrained", "alg1"): "#0072B2",
    ("retrained", "diffpca"): "#56B4E9",
}


def load_json(name: str):
    return json.loads((PAPER / "numbers" / name).read_text())


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def grid_pooled_rows() -> dict[str, dict[str, str]]:
    path = ROOT / "results" / "E9" / "joint_pooled_Qwen2.5-7B-Instruct.csv"
    with path.open() as fh:
        return {r["cond"]: r for r in csv.DictReader(fh) if r["pool"] == "all11"}


def capture_figure() -> None:
    data = load_json("b5_geometry.json")
    models = [
        ("Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
        ("Qwen3-8B", "Qwen3-8B"),
        ("DeepSeek-R1-Distill-Qwen-7B", "R1-Distill-7B"),
        ("GLM-Z1-9B-0414", "GLM-Z1-9B"),
    ]
    corpora = [("math", "MATH held-out"), ("mgsm", "MGSM cross-corpus")]
    pooled_color = "#7e4f98"
    fig, axes = plt.subplots(2, 4, figsize=(7.05, 3.15), sharex="col", sharey=True)

    for row, (corpus, row_name) in enumerate(corpora):
        for col, (model, model_name) in enumerate(models):
            ax = axes[row, col]
            d = data["models"][model][corpus]
            n_layers = int(d["n_layers"])
            xs = [k / (n_layers - 1) for k in range(n_layers)]
            alg = [d["alg1_by_layer"][str(k)] for k in range(n_layers)]
            dif = [d["diffpca_by_layer"][str(k)] for k in range(n_layers)]
            # Read the rank-matched control from the same E3 sweep as the summary.
            source = ROOT / "results" / "E3" / model / f"E3_rank_sweep_{corpus}_{model}_n7500.csv"
            with source.open() as fh:
                rows = [r for r in csv.DictReader(fh) if int(r["rank"]) == 10]
            curves = {}
            for method in ("alg1", "diffpca", "pooled"):
                selected = [r for r in rows if r["subspace"] == method]
                curves[method] = {int(r["layer"]): float(r["rho_perprompt"]) for r in selected}
                assert len(selected) == n_layers and set(curves[method]) == set(range(n_layers)), (model, corpus, method)
            for method, summary in (("alg1", alg), ("diffpca", dif)):
                assert all(abs(curves[method][k] - summary[k]) <= 1e-6 for k in range(n_layers)), (model, corpus, method)
            pooled = [curves["pooled"][k] for k in range(n_layers)]
            ax.plot(xs, pooled, color=pooled_color, linestyle="-.", label="Activation-PCA")
            ax.plot(xs, alg, color=ALG, label="Mean-SVD")
            ax.plot(xs, dif, color=DIFF, linestyle="--", label="DiffPCA")
            trough = d["alg1_trough"]
            tx, ty = trough["depth"], trough["rho"]
            ax.scatter([tx], [ty], s=26, facecolor="white", edgecolor=ALG,
                       linewidth=1.1, zorder=5)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0.18, 0.94)
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(axis="y", length=0)
            if row == 0:
                ax.set_title(model_name)
            if col == 0:
                ax.set_ylabel(f"{row_name}\ntranslation capture")
            if row == 1:
                ax.set_xlabel("Relative depth")
            else:
                ax.tick_params(labelbottom=False)

    handles = [
        Line2D([], [], color=ALG, label="Mean-SVD"),
        Line2D([], [], color=DIFF, linestyle="--", label="DiffPCA"),
        Line2D([], [], color=pooled_color, linestyle="-.", label="Activation-PCA"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.8,
               columnspacing=1.2, handletextpad=0.45)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.84, bottom=0.15,
                        wspace=0.16, hspace=0.11)
    fig.savefig(HERE / "fig_capture_rewrite.pdf")
    fig.savefig(HERE / "fig_capture_rewrite.png", dpi=220)
    plt.close(fig)


def probe_depth_figure() -> None:
    """Contrast fixed and retrained language readouts across model depth."""
    models = [
        ("Qwen2.5-7B-Instruct", "Qwen2.5"),
        ("Qwen3-8B", "Qwen3"),
        ("DeepSeek-R1-Distill-Qwen-7B", "R1-Distill"),
        ("GLM-Z1-9B-0414", "GLM-Z1"),
    ]
    styles = {"alg1": "-", "diffpca": "--"}
    probe_kinds = {"frozen", "retrained"}

    fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.1), sharex=True, sharey=True)
    for ax, (model, title) in zip(axes, models):
        path = (
            ROOT / "results" / "E10" / model
            / f"E10_probe_erasure_{model}.csv"
        )
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        rows = [
            r for r in rows
            if r["rank"] == "10"
            and float(r["lam"]) == 1.0
            and r["subspace"] in styles
            and r["probe_kind"] in probe_kinds
        ]
        max_layer = max(int(r["layer"]) for r in rows)
        floor = sum(float(r["acc_floor"]) for r in rows) / len(rows)
        ax.axhline(floor, color="#999999", linestyle=":", linewidth=0.9, zorder=1)
        for probe_kind in ("frozen", "retrained"):
            for subspace in ("alg1", "diffpca"):
                series = sorted(
                    (r for r in rows
                     if r["probe_kind"] == probe_kind and r["subspace"] == subspace),
                    key=lambda r: int(r["layer"]),
                )
                xs = [int(r["layer"]) / max_layer for r in series]
                ys = [float(r["acc"]) for r in series]
                ax.plot(xs, ys, color=PROBE_COLORS[(probe_kind, subspace)],
                        linestyle=styles[subspace], linewidth=1.3,
                        marker="o" if probe_kind == "frozen" else "s",
                        markersize=2.5, markevery=max(1, len(series) // 6),
                        alpha=0.95, zorder=2)

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.025)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("Relative depth")
        ax.grid(axis="y", color="#e1e1e1", linewidth=0.45, alpha=0.8)
    axes[0].set_ylabel("Probe accuracy")

    handles = [
        Line2D([], [], color=PROBE_COLORS[("frozen", "alg1")], linestyle="-",
               marker="o", markersize=3, label="Mean-SVD, frozen"),
        Line2D([], [], color=PROBE_COLORS[("frozen", "diffpca")], linestyle="--",
               marker="o", markersize=3, label="DiffPCA, frozen"),
        Line2D([], [], color=PROBE_COLORS[("retrained", "alg1")], linestyle="-",
               marker="s", markersize=3, label="Mean-SVD, retrained"),
        Line2D([], [], color=PROBE_COLORS[("retrained", "diffpca")], linestyle="--",
               marker="s", markersize=3, label="DiffPCA, retrained"),
        Line2D([], [], color="#999999", linestyle=":", label="Permutation floor"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.01), handlelength=2.0,
               columnspacing=0.7, handletextpad=0.35)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.79, bottom=0.25, wspace=0.16)
    fig.savefig(HERE / "fig_probe_depth_rewrite.pdf")
    fig.savefig(HERE / "fig_probe_depth_rewrite.png", dpi=220)
    plt.close(fig)


def grid_heatmap_figure() -> None:
    """Two complete slices of the 37-arm grid, with a shared linear color scale."""
    rows = {k: v for k, v in grid_pooled_rows().items() if k != 'clean'}
    assert len(rows) == 37
    specs = [('alg1', op) for op in ['proj', 'leace', 'wiener']] + [
        ('diffpca', op) for op in ['proj', 'leace', 'wiener']]
    ranks, doses = [1, 4, 10, 32], [50, 100, 150, 200]
    matrices = [np.full((6, 4), np.nan), np.full((6, 4), np.nan)]
    keys = {}
    used = set()
    for panel in range(2):
        for i, (basis, op) in enumerate(specs):
            for j in range(4):
                if panel == 0 and (basis, op) == ('alg1', 'wiener'):
                    continue  # This operator has intrinsic rank, not a rank parameter.
                rank, dose = (ranks[j], 100) if panel == 0 else (10, doses[j])
                key = f'e9_{basis}_{op}_r{rank}_d{dose:03d}'
                if key in rows:
                    matrices[panel][i, j] = 100 * float(rows[key]['d_joint_vs_clean'])
                    keys[panel, i, j] = key
                    used.add(key)
    assert used == set(rows), set(rows) - used
    assert sum(np.isfinite(m).sum() for m in matrices) == 42  # Five shared references.
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#ffffff')
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.0))
    marks = {'e9_alg1_proj_r1_d100': 'J', 'e9_alg1_proj_r10_d050': 'A',
             'e9_alg1_proj_r10_d100': 'R'}
    for panel, ax in enumerate(axes):
        mat = matrices[panel]
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect='auto')
        for i in range(6):
            for j in range(4):
                val = mat[i, j]
                if not np.isfinite(val):
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor='#fafafa',
                                           edgecolor='#dddddd', hatch='///', linewidth=.4))
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=7, color='#666666',
                            bbox=dict(facecolor='#fafafa', edgecolor='none', pad=.4))
                    continue
                color = 'white' if abs(val) > 35 else '#222222'
                mark = marks.get(keys[panel, i, j])
                label = f'{val:+.1f}' + (rf'$^{{\mathrm{{{mark}}}}}$' if mark else '')
                ax.text(j, i, label, ha='center', va='center', fontsize=8, color=color)
        ax.axhline(2.5, color='white', linewidth=3)
        ax.set_xticks(range(4), ['1', '4', '10', '32'] if panel == 0 else ['0.5', '1', '1.5', '2'])
        ax.set_yticks(range(6), ['Projection', 'Oblique', 'Wiener*'] * 2 if panel == 0 else [])
        if panel == 0:
            ax.set_yticklabels(['Projection', 'Oblique', 'Wiener*', 'Projection', 'Oblique', 'Wiener'])
        ax.tick_params(length=0, labelsize=8)
        ax.set_xlabel('Subspace rank' if panel == 0 else 'Target edit energy / reference energy', fontsize=8)
        ax.set_title('(a) Rank sweep\nReference coefficients' if panel == 0 else
                     '(b) Dose sweep\nRank 10; intrinsic rank for Wiener*', fontsize=9, pad=9)
        for spine in ax.spines.values(): spine.set_visible(False)
    for y, label in [(0.75, 'Mean-SVD'), (.25, 'DiffPCA')]:
        axes[0].text(-.31, y, label, transform=axes[0].transAxes, rotation=90,
                     ha='center', va='center', fontsize=9, fontweight='bold')
    panel_center = (.19 + .985) / 2
    fig.suptitle('Joint-accuracy change from no intervention', fontsize=11,
                 x=panel_center, y=.985)
    fig.text(panel_center, .916, 'Qwen2.5-7B · MGSM · 11-language mean',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.19, right=.985, top=.78, bottom=.32, wspace=.12)
    cax = fig.add_axes([.28, .195, .60, .027])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-60, -30, 0, 30, 60])
    cb.set_label('Joint-accuracy change (percentage points)', fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    fig.text(.5, .09, 'J  Best joint accuracy     A  Best answer accuracy     R  Reference configuration',
             ha='center', fontsize=8)
    fig.text(.5, .045, '* Mean-SVD/Wiener has no rank knob; its reference appears at dose 1.  N/A: unavailable.',
             ha='center', fontsize=7.5)
    fig.savefig(HERE / 'fig_grid_heatmap_rewrite.pdf')
    fig.savefig(HERE / 'fig_grid_heatmap_rewrite.png', dpi=220)
    plt.close(fig)


def grid_figure() -> None:
    grid_heatmap_figure()
    rows = [r for cond, r in grid_pooled_rows().items() if cond != "clean"]

    # Main-text figure: the same cells in the outcome plane. The broken y-axis prevents
    # the three rank-32 failures from compressing the scientifically relevant region.
    fig = plt.figure(figsize=(3.35, 3.15))
    right = fig.add_gridspec(2, 1, height_ratios=(3.5, 0.8), hspace=0.07)
    ax_main = fig.add_subplot(right[0])
    ax_tail = fig.add_subplot(right[1], sharex=ax_main)
    outcome_colors = {"joint": "#009E73", "accuracy": "#E69F00", "other": "#777777"}
    outcome_markers = {"joint": "^", "accuracy": "D", "other": "o"}
    for r in rows:
        x = 100 * float(r["d_fid_strict_vs_clean"])
        y = 100 * float(r["d_correct_vs_clean"])
        joint = 100 * float(r["d_joint_vs_clean"])
        category = "joint" if joint > 0 else ("accuracy" if y > 0 else "other")
        for ax in (ax_main, ax_tail):
            ax.scatter(x, y, s=25, marker=outcome_markers[category], color=outcome_colors[category],
                       edgecolor="white", linewidth=0.35, zorder=3)

    for ax in (ax_main, ax_tail):
        ax.axvline(0, color="#777777", linewidth=0.65, zorder=1)
        ax.grid(color="#e0e0e0", linewidth=0.45, alpha=0.7, zorder=0)
        ax.set_xlim(-58, 4)
    ax_main.axhline(0, color="#777777", linewidth=0.65, zorder=1)
    ax_main.set_ylim(-16.5, 5.5)
    ax_main.set_yticks([-15, -10, -5, 0, 5])
    ax_tail.set_ylim(-53.5, -38)
    ax_tail.set_yticks([-50, -40])
    ax_tail.set_xticks([-50, -25, 0])
    ax_main.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_tail.set_xlabel("Response-language fidelity change (points)")
    ax_main.set_title("Accuracy and fidelity outcomes", pad=10)

    ax_main.spines["bottom"].set_visible(False)
    ax_tail.spines["top"].set_visible(False)
    d = 0.012
    break_kw = {"color": "#333333", "clip_on": False, "linewidth": 0.75}
    ax_main.plot((-d, +d), (-d, +d), transform=ax_main.transAxes, **break_kw)
    ax_main.plot((1 - d, 1 + d), (-d, +d), transform=ax_main.transAxes, **break_kw)
    ax_tail.plot((-d, +d), (1 - d, 1 + d), transform=ax_tail.transAxes, **break_kw)
    ax_tail.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_tail.transAxes, **break_kw)

    annotations = {
        "e9_alg1_proj_r10_d050": ("accuracy best", (-36, 4.3)),
        "e9_alg1_proj_r1_d100": ("joint best", (-12, 3.9)),
    }
    for r in rows:
        if r["cond"] not in annotations:
            continue
        label, xytext = annotations[r["cond"]]
        xy = (100 * float(r["d_fid_strict_vs_clean"]),
              100 * float(r["d_correct_vs_clean"]))
        ax_main.annotate(label, xy, xytext=xytext, textcoords="data", fontsize=7.2,
                         arrowprops={"arrowstyle": "-", "lw": 0.5,
                                     "color": "#555555"})

    handles = [
        Line2D([], [], marker="^", linestyle="none", color=outcome_colors["joint"],
               label="joint gain", markersize=4.4),
        Line2D([], [], marker="D", linestyle="none", color=outcome_colors["accuracy"],
               label="accuracy gain\nonly", markersize=4.4),
        Line2D([], [], marker="o", linestyle="none", color=outcome_colors["other"],
               label="no accuracy\ngain", markersize=4.4),
    ]
    fig.subplots_adjust(left=0.20, right=0.985, top=0.91, bottom=0.29)
    panel_b = ax_tail.get_position()
    fig.text(0.035, 0.61, "Accuracy change (points)", rotation=90,
             ha="center", va="center")
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.57, 0.015), handletextpad=0.35,
               columnspacing=1.1, labelspacing=0.55)
    fig.savefig(HERE / "fig_grid_rewrite.pdf")
    fig.savefig(HERE / "fig_grid_rewrite.png", dpi=220)
    plt.close(fig)


def probe_table() -> None:
    data = load_json("b6_probes.json")["models"]
    models = [
        ("Qwen2.5-7B-Instruct", "Qwen2.5"),
        ("Qwen3-8B", "Qwen3"),
        ("DeepSeek-R1-Distill-Qwen-7B", "R1-Distill"),
        ("GLM-Z1-9B-0414", "GLM-Z1"),
    ]
    rows = []
    for key, name in models:
        h = data[key]["headline_rank10_lam1_at_trough"]
        cells = [f'{h[s]["frozen"]:.3f}/{h[s]["retrained"]:.3f}'
                 for s in ("alg1", "diffpca", "pooled")]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    write(
        HERE / "tab_probe_rewrite.tex",
        """\\begin{tabular}{@{}lccc@{}}
\\toprule
Model & \\AlgOne{} & \\DiffPCA{} & \\Pooled{} \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def grid_table() -> None:
    data = grid_pooled_rows()

    def pct(cond: str, key: str) -> str:
        return f"{100 * float(data[cond][key]):.2f}"

    selected = [
        (r"\shortstack[l]{\AlgOne{}, rank 10\\projection, dose 0.5}",
         "e9_alg1_proj_r10_d050"),
        (r"\shortstack[l]{\AlgOne{}, rank 1\\projection, dose 1}",
         "e9_alg1_proj_r1_d100"),
    ]
    rows = [
        "No intervention & "
        + " & ".join(pct("clean", k) for k in ("correct", "fid_strict", "joint"))
        + r" \\"
    ]
    for label, cond in selected:
        rows.extend([
            r"\addlinespace[3pt]",
            label + " & "
            + " & ".join(pct(cond, k) for k in ("correct", "fid_strict", "joint"))
            + r" \\",
        ])
    write(
        HERE / "tab_grid_rewrite.tex",
        """\\begin{tabular}{@{}lrrr@{}}
\\toprule
\\multicolumn{1}{l}{Setting} & \\multicolumn{1}{c}{Accuracy} &
  \\multicolumn{1}{c}{Fidelity} & \\multicolumn{1}{c}{Joint} \\\\
& \\multicolumn{1}{c}{(\\%) $\\uparrow$} & \\multicolumn{1}{c}{(\\%) $\\uparrow$} &
  \\multicolumn{1}{c}{(\\%) $\\uparrow$} \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def trough_table() -> None:
    data = load_json("b5_geometry.json")["models"]
    specs = [
        ("Qwen2.5-7B-Instruct", "Qwen2.5"),
        ("Qwen3-8B", "Qwen3"),
        ("DeepSeek-R1-Distill-Qwen-7B", "R1"),
        ("GLM-Z1-9B-0414", "GLM-Z1"),
    ]
    rows = []
    for key, label in specs:
        for corpus, macro in (("math", r"\MATHD{}"), ("mgsm", r"\MGSM{}")):
            d = data[key][corpus]
            t = d["alg1_trough"]
            layer = str(t["layer"])
            rows.append(
                f"{label}/{macro} & {t['rho']:.3f} & "
                f"{d['diffpca_by_layer'][layer]:.3f} & {layer}/{t['depth']:.2f} & "
                f"{d['pearson_alg1_diffpca']:.4f} " + r"\\"
            )
    write(
        HERE / "tab_trough_rewrite.tex",
        """\\begin{tabular}{@{}lrrrr@{}}
\\toprule
Model/data & \\AlgOne{} & \\DiffPCA{} & $k$/depth & $r_{\\mathrm{P}}$ \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def prompt_joint_delta(path: Path, arm: str, languages: list[str] | None = None) -> dict:
    """Strict per-response prompt deltas on the arm's own or a supplied language basis."""
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for row in rows:
        by.setdefault(row["cond"], {})[row["lang"]] = row
    basis = sorted(set(by[arm]) & set(by["clean"])) if languages is None else languages
    assert set(basis) <= set(by[arm]) & set(by["clean"])
    return {
        metric: statistics.fmean(
            float(by[arm][lang][column]) - float(by["clean"][lang][column])
            for lang in basis
        )
        for metric, column in (
            ("accuracy", "accuracy"),
            ("fidelity", "fidelity_strict"),
            ("joint", "joint"),
        )
    }


def prompt_table() -> None:
    """Appendix prompt results on the nine-language basis used in the main table."""
    languages = ['bn', 'de', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th']
    data = load_json("b9_prompt_control.json")["models"]
    specs = [
        ('Qwen2.5', 'Qwen2.5-7B-Instruct', None),
        ('Qwen3 (thinking)', None, ROOT / 'results/E8/recovered/Qwen3-8B_think_joint.csv'),
        ('Qwen3 (non-thinking)', None, ROOT / 'results/E8/recovered/Qwen3-8B_nothink_joint.csv'),
        ('R1-Distill', 'DeepSeek-R1-Distill-Qwen-7B', None),
        ('GLM-Z1', None, ROOT / 'results/E8/promptctl_joint_GLM-Z1-9B-0414_glm-v2-seed0.csv'),
    ]
    lines = [r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
             r'& \multicolumn{3}{c}{Prompt-language switch} & \multicolumn{3}{c}{Translate-test} \\',
             r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
             r'Model & Accuracy & Fidelity & Joint & Accuracy & Fidelity & Joint \\']
    for pivot, label in [('en', 'English'), ('zh', 'Chinese')]:
        lines += [r'\midrule', r'\multicolumn{7}{@{}l}{\textbf{' + label + r'}} \\', r'\addlinespace[2pt]']
        for model, tag, path in specs:
            values = []
            for arm in [f'{pivot}_cot', f'{pivot}_question']:
                if path:
                    metrics = prompt_joint_delta(path, arm, languages)
                else:
                    perlang = data[tag]['per_language']
                    assert set(languages) <= set(perlang[arm]) & set(perlang['clean'])
                    metrics = {m: statistics.fmean(perlang[arm][lang][m] - perlang['clean'][lang][m]
                                                   for lang in languages)
                               for m in ['accuracy', 'fidelity', 'joint']}
                values += [100 * metrics[m] for m in ['accuracy', 'fidelity', 'joint']]
            assert all(-100 <= v <= 100 for v in values)
            lines.append(model + ' & ' + ' & '.join(f'${v:+.2f}$' for v in values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write(HERE / 'tab_prompt_rewrite.tex', '\n'.join(lines))

    # Absolute joint scores make the native-prompt reference visible for both seeds.
    columns = []
    conditions = ['clean', 'en_cot', 'en_cot_target', 'zh_cot', 'zh_cot_target']
    for seed in [0, 1]:
        path = ROOT / f'results/E8/promptctl_joint_GLM-Z1-9B-0414_glm-v2-seed{seed}.csv'
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        column = {}
        for cond in conditions:
            selected = [r for r in rows if r['cond'] == cond and r['lang'] in languages]
            assert len(selected) == 9 and {r['lang'] for r in selected} == set(languages)
            assert all(int(r['n']) == 250 for r in selected)
            column[cond] = 100 * statistics.fmean(float(r['joint']) for r in selected)
        columns.append(column)
    labels = ['Native prompt', 'English switch', r'English + target instruction',
              'Chinese switch', r'Chinese + target instruction']
    target = [r'\begin{tabular}{@{}lrr@{}}', r'\toprule',
              r'Prompt & Seed 0 & Seed 1 \\', r'\midrule']
    for cond, label in zip(conditions, labels):
        target.append(label + ' & ' + ' & '.join(f'{c[cond]:.2f}' for c in columns) + r' \\')
    target += [r'\bottomrule', r'\end{tabular}']
    write(HERE / 'tab_prompt_target_appendix.tex', '\n'.join(target))


def intervention_prompt_table() -> None:
    """Compare all methods on the nine languages shared by both pivot arms.

    Published accuracies are transcribed in a separately sourced JSON. Our marginal
    metrics come from per-language aggregates; joint scores require item-based tables.
    Every delta uses the baseline run belonging to that method and generation protocol.
    """
    published = load_json("zhao2025_mgsm_table1.json")
    joint_data = load_json("b9_prompt_control.json")["models"]
    languages = sorted(set(published["language_order"]) - {"en", "zh"})
    specs = [
        ("Qwen2.5-7B-Instruct", "Qwen2.5", "Qwen2.5-7B-Instruct", None,
         "Qwen2.5-7B-Instruct"),
        ("Qwen3-8B_think", "Qwen3 (thinking)", None,
         ROOT / "results/E8/recovered/Qwen3-8B_think_joint.csv", "Qwen3-8B_think"),
        ("Qwen3-8B_nothink", "Qwen3 (no thinking)", None,
         ROOT / "results/E8/recovered/Qwen3-8B_nothink_joint.csv", None),
        ("DeepSeek-R1-Distill-Qwen-7B", "R1-Distill",
         "DeepSeek-R1-Distill-Qwen-7B", None, "DeepSeek-R1-Distill-Qwen-7B"),
        ("GLM-Z1-9B-0414_glm-v2-seed0", "GLM-Z1", None,
         ROOT / "results/E8/promptctl_joint_GLM-Z1-9B-0414_glm-v2-seed0.csv",
         "GLM-Z1-9B-0414"),
    ]
    records, tex_rows = {}, []
    for tag, label, joint_tag, joint_path, published_tag in specs:
        path = ROOT / "results/E8" / f"promptctl_agg_{tag}.csv"
        by = {}
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row["lang"] != "lang":
                    by.setdefault(row["cond"], {})[row["lang"]] = row
        assert set(languages) == set(by["en_cot"]) & set(by["zh_cot"])
        record = {"aggregate_source": str(path.relative_to(ROOT)),
                  "joint_source": (str(joint_path.relative_to(ROOT)) if joint_path else
                                   "paper/workshop/numbers/b9_prompt_control.json"),
                  "pivots": {},
                  "translate_test_accuracy_delta": {}}
        for pivot in ("en", "zh"):
            arm = f"{pivot}_cot"
            if joint_path:
                metrics = {k: 100 * v for k, v in
                           prompt_joint_delta(joint_path, arm, languages).items()}
            else:
                per_language = joint_data[joint_tag]["per_language"]
                metrics = {
                    metric: 100 * statistics.fmean(
                        per_language[arm][lang][metric]
                        - per_language["clean"][lang][metric]
                        for lang in languages
                    )
                    for metric in ("accuracy", "fidelity", "joint")
                }
            record["pivots"][pivot] = metrics
            record["translate_test_accuracy_delta"][pivot] = 100 * statistics.fmean(
                float(by[f"{pivot}_question"][lang]["acc"]) - float(by["clean"][lang]["acc"])
                for lang in languages
            )
        record["zhao_accuracy_delta"] = None
        record["zhao_fidelity_delta"] = None
        if published_tag is not None:
            summary = published["models"][published_tag]["reported_average"]
            record["zhao_accuracy_delta"] = (
                summary["intervention_accuracy"] - summary["clean_accuracy"]
            )
            record["zhao_fidelity_delta"] = (
                summary["intervention_fidelity"] - summary["clean_fidelity"]
            )
        records[tag] = record

        def pp(value):
            return "--" if value is None else f"${value:+.2f}$"

        cells = [record["zhao_accuracy_delta"], record["zhao_fidelity_delta"]]
        cells += [record["pivots"][p][m] for p in ("en", "zh")
                  for m in ("accuracy", "fidelity")]
        tex_rows.append(label + " & " + " & ".join(pp(v) for v in cells) + r" \\")

    write(HERE / "tab_intervention_prompt_rewrite.tex", r"""\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{Causal intervention\textsuperscript{*}} &
  \multicolumn{2}{c}{English switch} & \multicolumn{2}{c}{Chinese switch} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Model & Accuracy & Fidelity & Accuracy & Fidelity & Accuracy & Fidelity \\
\midrule
""" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}")
    write(PAPER / "numbers/intervention_prompt_comparison.json", json.dumps({
        "generated_by": "paper/workshop/figures/make_rewrite_assets.py",
        "units": "percentage-point change from each protocol's own baseline",
        "prompt_switch_languages": languages,
        "published_source": published["source_url"],
        "published_table": "Table 1; reported 11-language accuracy and fidelity averages",
        "fidelity_conventions": {
            "prompt_pivots": "strict; unidentified responses count as failures",
            "joint": "strict for every score",
        },
        "models": records,
    }, indent=2))


if __name__ == "__main__":
    capture_figure()
    probe_depth_figure()
    grid_figure()
    probe_table()
    grid_table()
    trough_table()
    prompt_table()
    intervention_prompt_table()
