"""Extract final-token activations for the **MGSM** problems, in all 11 languages.

The MATH activation cache (`acts_<tag>_n7500_stratified-s0.pt`) is what every subspace in this
repo is fit on. This script builds its counterpart on the *benchmark* corpus, so
`scripts/e3_rank_sweep.py --mode mgsm` can measure the captured fraction of translation vectors
the subspace has never seen — a cross-corpus read, and the thing that separates "the language
subspace of Qwen2.5-7B" from "the language subspace of our parallel MATH set".

## Two deliberate choices

**0-shot prompts.** MGSM evaluation runs 8-shot, but the exemplars are *localized per language*,
so an 8-shot prompt prepends a long language-consistent block to every item. The final-token
residual would then be dominated by the exemplars rather than by the problem statement, inflating
the language signal for a reason that has nothing to do with the question being asked. 0-shot
keeps the activation about the problem. The prompt is otherwise built by the same
`build_mgsm_prompt` path evaluation uses, through the same chat template (`build_chat_inputs`),
so the capture point matches the rest of the repo.

**The test split, in dataset order, with no subsampling by default.** MGSM is fully parallel: item
`i` in `bn` is the translation of item `i` in `en`. Translation vectors are therefore defined by
position, and the script asserts every language returned the same count rather than trusting it.

Writes `data/activations/mgsm_acts_<tag>_nshot<k>.pt` = `{lang: (n, n_layers, d)}` fp32/CPU, plus
a `.meta.json` sidecar recording the item ids so a later run can prove it used the same problems.

Environment knobs (set by scripts/prajna/e3_prajna.sbatch; the job is --export=NONE):

  LRD_MODEL         HF model id
  E3M_SPLIT         MGSM split                                (default test)
  E3M_NSHOT         few-shot exemplars                        (default 0 — see above)
  E3M_N             cap the item count, "" for all            (default all)
  E3M_BATCH         extraction batch size                     (default 16)

Usage:
    python scripts/e3_mgsm_acts.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch

_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
             if (p / "src" / "config.py").exists())
sys.path.insert(0, str(_root / "src"))

from benchmarks import load_benchmark                                    # noqa: E402
from config import ACTS_DIR, Config, get_device                          # noqa: E402
from extract import (extract_last_token_per_layer, load_model_and_tokenizer,  # noqa: E402
                     print_massive_activation_report)


def main() -> int:
    t0 = time.time()
    cfg = Config(model_name=os.environ.get("LRD_MODEL", Config.model_name))
    tag = cfg.model_name.split("/")[-1]
    langs = list(cfg.langs)
    split = os.environ.get("E3M_SPLIT", "test")
    n_shot = int(os.environ.get("E3M_NSHOT", "0"))
    n_env = os.environ.get("E3M_N", "").strip()
    n = int(n_env) if n_env else None
    batch = int(os.environ.get("E3M_BATCH", "16"))

    out = ACTS_DIR / f"mgsm_acts_{tag}_nshot{n_shot}.pt"
    if out.exists():
        print(f"MGSM activation cache already present: {out} — nothing to do.")
        return 0

    print(f"loading MGSM ({split}, {n_shot}-shot) for {len(langs)} languages", flush=True)
    items = {l: load_benchmark("mgsm", l, split, cfg=cfg, n=n, seed=cfg.seed, n_shot=n_shot)
             for l in langs}
    counts = {l: len(v) for l, v in items.items()}
    if len(set(counts.values())) != 1:
        print(f"ERROR: MGSM is not parallel across languages under this selection: {counts}",
              file=sys.stderr)
        return 3
    n_items = next(iter(counts.values()))
    print(f"  {n_items} parallel problems per language", flush=True)

    device = get_device()
    model, tok = load_model_and_tokenizer(cfg, device)

    acts: dict[str, torch.Tensor] = {}
    for l in langs:
        t_l = time.time()
        acts[l] = extract_last_token_per_layer(
            model, tok, [it.prompt for it in items[l]], device, batch_size=batch
        )
        print(f"[{time.time()-t0:7.1f}s] {l}: {tuple(acts[l].shape)} in {time.time()-t_l:.1f}s",
              flush=True)

    print_massive_activation_report(acts)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(acts, out)
    with open(out.with_suffix(".meta.json"), "w") as f:
        json.dump({
            "model": cfg.model_name,
            "corpus": "mgsm",
            "split": split,
            "n_shot": n_shot,
            "n_items": n_items,
            "langs": langs,
            "item_ids": {l: [it.meta.get("id") for it in items[l]] for l in langs},
            "note": "0-shot by default: MGSM's few-shot exemplars are localized, so an 8-shot "
                    "prompt would make the final-token residual mostly about the exemplar block.",
        }, f, indent=2)
    print(f"\nwrote {out} and its sidecar  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
