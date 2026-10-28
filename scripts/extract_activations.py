#!/usr/bin/env python3
"""Extract local MATH activation caches for the paper pipeline."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--n', type=int, default=7500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    from config import Config, set_seeds
    from extract import acts_cache_path, load_or_extract_acts
    set_seeds(args.seed)
    cfg = Config(model_name=args.model, n_samples=args.n, seed=args.seed)
    load_or_extract_acts(cfg, batch_size=args.batch_size, allow_download=False)
    print(acts_cache_path(cfg))


if __name__ == '__main__':
    main()
