#!/usr/bin/env python3
"""Print the paper's headline results using only the standard library."""
import csv
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / 'results/E9/joint_pooled_Qwen2.5-7B-Instruct.csv'
    with path.open() as stream:
        rows = {r['cond']: r for r in csv.DictReader(stream) if r['pool'] == 'all11'}
    choices = [('No intervention', 'clean'),
               ('Highest answer accuracy', 'e9_alg1_proj_r10_d050'),
               ('Highest joint accuracy', 'e9_alg1_proj_r1_d100')]
    print('BeyondAccuracy | Qwen2.5-7B | MGSM | 11 languages x 250 problems')
    print(f"{'Configuration':<27} {'Accuracy':>10} {'Fidelity':>10} {'Joint':>10}")
    for label, key in choices:
        row = rows[key]
        scores = [100 * float(row[k]) for k in ['correct', 'fid_strict', 'joint']]
        print(f'{label:<27} ' + ' '.join(f'{v:9.2f}%' for v in scores))
    print('\nJoint = correct AND in the requested language, on the same response.')
    print('Descriptive grid selections; see docs/reproduction.md for scope and protocol.')


if __name__ == '__main__':
    main()
