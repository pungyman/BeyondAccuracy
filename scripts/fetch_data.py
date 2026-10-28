#!/usr/bin/env python3
"""Fetch aligned MATH translations from a pinned public upstream revision.

Downloads data only; does not execute code or require private cache credentials.
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

REVISION = '059b57d8895a7ce5b525431b0abbff620ebd33f1'
LANGS = 'bn de en es fr ja ru sw te th zh'.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default=REVISION, help='Upstream 40-character Git commit')
    parser.add_argument('--languages', nargs='+', choices=LANGS, default=LANGS)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/parallel_multilingual_math')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('--revision must be a full lowercase Git commit SHA')
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / 'upstream-revision.txt'
    if marker.exists() and marker.read_text().strip() != args.revision:
        parser.error('Output contains another revision; choose a fresh --output directory')
    for lang in args.languages:
        target = args.output / f'{lang}.json'
        if target.exists() and not marker.exists():
            parser.error(f'{target} exists without download provenance; use a fresh --output directory')
        if target.exists():
            print(f'Already present: {target.name}')
            continue
        url = f'https://raw.githubusercontent.com/MuyuenLP/Language-Reasoning-Disentangle/{args.revision}/data/parallel_multilingual_math/{lang}.json'
        print(f'Downloading {lang}.json', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        records = json.loads(content)
        if not isinstance(records, list) or not records or not all('translated_prompt' in r for r in records):
            raise ValueError(f'Unexpected schema for {lang}')
        temp = target.with_suffix('.json.part')
        temp.write_bytes(content)
        temp.replace(target)
        marker.write_text(args.revision + '\n')
    print('Done. See NOTICE.md for upstream attribution and dataset terms.')


if __name__ == '__main__':
    main()
