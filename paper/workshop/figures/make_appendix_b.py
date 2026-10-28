"""Generate Appendix B probe assets from the committed E10 results.

Run from any directory. Only appendix assets are written; no experiment is rerun.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperstyle

paperstyle.assert_serif_available()
paperstyle.apply()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODELS = [
    ('Qwen2.5-7B-Instruct', 'Qwen2.5'),
    ('Qwen3-8B', 'Qwen3'),
    ('DeepSeek-R1-Distill-Qwen-7B', 'R1-Distill'),
    ('GLM-Z1-9B-0414', 'GLM-Z1'),
]
METHODS = [
    ('alg1', 'Mean-SVD', '#2465a6', 'o'),
    ('diffpca', 'DiffPCA', '#c06016', 's'),
    ('pooled', 'Activation-PCA', '#7e4f98', '^'),
    ('random', 'Random', '#44804a', 'D'),
]


def load(tag):
    path = ROOT / 'results' / 'E10' / tag / f'E10_probe_erasure_{tag}.csv'
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ('layer', 'rank', 'trough_layer'):
            row[key] = int(row[key])
        for key in ('acc', 'acc_floor', 'lam'):
            row[key] = float(row[key])
    keys = [(r['layer'], r['rank'], r['lam'], r['subspace'], r['probe_kind']) for r in rows]
    assert len(keys) == len(set(keys)), f'Duplicate experiment cells: {tag}'
    assert len({r['trough_layer'] for r in rows}) == 1
    return rows


def pick(rows, **criteria):
    selected = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    assert len(selected) == 1, (criteria, len(selected))
    return selected[0]


def main():
    table_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.7), sharey=True)
    for col, (tag, label) in enumerate(MODELS):
        rows = load(tag)
        layer = rows[0]['trough_layer']
        anchor = [r for r in rows if r['layer'] == layer]
        clean = pick(anchor, subspace='clean')
        assert clean['floor_measured'] == 'True'
        assert clean['acc'] == 1
        cells = []
        for method, _, _, _ in METHODS:
            pair = [pick(anchor, subspace=method, rank=10, lam=1.0, probe_kind=kind)['acc']
                    for kind in ('frozen', 'retrained')]
            cells.append(f'{pair[0]:.3f} / {pair[1]:.3f}')
        minimum = min(r['acc'] for r in rows if r['probe_kind'] == 'retrained' and r['subspace'] != 'clean')
        table_rows.append(f'{label} & {layer} & ' + ' & '.join(cells) +
                          f" & {clean['acc_floor']:.3f} & {minimum:.4f} " + r'\\')
        for row_idx, xkey in enumerate(('rank', 'lam')):
            ax = axes[row_idx, col]
            for method, _, color, marker in METHODS:
                for kind, style in [('frozen', '-'), ('retrained', '--')]:
                    vals = sorted([r for r in anchor if r['subspace'] == method and r['probe_kind'] == kind
                                   and (r['lam'] == 1.0 if xkey == 'rank' else r['rank'] == 10)], key=lambda r: r[xkey])
                    expected = (5 if method == 'alg1' else 8) if xkey == 'rank' else 5
                    assert len(vals) == expected, (tag, method, xkey, len(vals))
                    ax.plot([r[xkey] for r in vals], [r['acc'] for r in vals], linestyle=style,
                            color=color if kind == 'frozen' else '0.15',
                            zorder=2 if kind == 'frozen' else 3,
                            marker=marker if kind == 'frozen' else None,
                            markersize=3, linewidth=1.1, alpha=0.95)
            ax.axhline(clean['acc_floor'], color='0.5', linestyle=':', linewidth=1)
            ax.set_ylim(-.025, 1.055)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.tick_params(labelsize=8, length=2)
            ax.grid(axis='y', alpha=.17)
            ax.spines[['top', 'right']].set_visible(False)
            if row_idx == 0:
                ax.set_xscale('log')
                ax.set_xticks([1, 3, 10, 30, 100], labels=['1', '3', '10', '30', '100'])
                ax.minorticks_off()
                ax.set_xlabel('Removed rank', fontsize=9)
                ax.set_title(f'{label} (layer {layer})', fontsize=10)
            else:
                ax.set_xticks([0, .25, .5, .75, 1], labels=['0', '.25', '.5', '.75', '1'])
                ax.set_xlabel(r'Projection strength $\lambda$', fontsize=9)
            if col == 0:
                ax.set_ylabel('Probe accuracy', fontsize=9)
    method_handles = [Line2D([], [], color=c, marker=m, markersize=4, linewidth=1.1, label=l)
                      for _, l, c, m in METHODS]
    readout_handles = [Line2D([], [], color='0.2', linestyle=s, label=l)
                       for s, l in [('-', 'Frozen'), ('--', 'Retrained'), (':', 'Permutation reference')]]
    fig.legend(handles=method_handles, ncol=4, loc='lower center', bbox_to_anchor=(.5, .065),
               fontsize=9, frameon=False)
    fig.legend(handles=readout_handles, ncol=3, loc='lower center', bbox_to_anchor=(.5, 0),
               fontsize=9, frameon=False)
    fig.subplots_adjust(left=.075, right=.985, top=.91, bottom=.24, wspace=.16, hspace=.56)
    fig.savefig(HERE / 'fig_probe_sweeps_appendix.pdf')
    fig.savefig(HERE / 'fig_probe_sweeps_appendix.png', dpi=220)
    plt.close(fig)
    (HERE / 'tab_probe_appendix.tex').write_text(
        '\\begin{tabular}{@{}lrccccrr@{}}\n\\toprule\n'
        'Model & $k$ & \\AlgOne{} & \\DiffPCA{} & \\Pooled{} & \\Rand{} & Perm. & Min. retr. \\\\\n'
        '\\midrule\n' + '\n'.join(table_rows) + '\n\\bottomrule\n\\end{tabular}\n')
    print('Generated Appendix B figure and table from all four models; cell checks passed.')


if __name__ == '__main__':
    main()
