"""E5 benchmark loaders + a unified record schema (design: docs/modules/benchmarks.md §2).

Three benchmarks, one `BenchItem` schema so the eval harness and scorers stay benchmark-agnostic:

- **MGSM** (`juletxara/mgsm`) — grade-school math, numeric generative CoT, fully parallel over
  our 11 languages. Native few-shot exemplars come from the dataset's own `train` split.
- **M-MMLU** (`CohereLabs/Global-MMLU`) — 4-way MC (log-likelihood over A–D), 10 of our 11 (no `th`).
- **XWinograd** (`Muennighoff/xwinograd`) — 2-way cloze (log-likelihood), 5 of our 11
  (`en fr ru zh ja`; the dataset's `jp` subset is normalized to `ja` on load).

Eval-language policy (E5 §3): each benchmark is evaluated on the intersection of the languages the
HF dataset provides and `cfg.langs` (our construction languages).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from config import REPO, Config
from prompts import build_mgsm_prompt, build_mmlu_prompt, build_xwinograd_continuations

BENCH_CACHE_DIR = REPO / "data" / "benchmarks"   # git-ignored; HF datasets cache_dir

_ALL_TASKS = ("mgsm", "mmlu", "xwinograd")

# `E5_TASKS=mgsm` narrows the run to a subset of the three benchmarks (E7 runs MGSM-only: every E7
# headline — H2's low-resource edge, H4/H5, G0, the Zhao Table-1 map — reads MGSM, and the retuned
# rank sweep is already MGSM-gated). Resolved HERE, not per-consumer: nb08 and e6_phase2_behavioral
# carried their own `os.environ["E5_TASKS"]` hook while nb06/nb07 did not, so an "MGSM-only" run
# still ground through the full M-MMLU + XWinograd set in P1/replication. One point of truth, no drift.
# A typo MUST fail loudly at import: filtering silently would leave TASKS empty and every phase would
# then evaluate nothing at all, which reads exactly like a fast, successful run.
TASKS = _ALL_TASKS
if os.environ.get("E5_TASKS"):
    _want = tuple(t.strip() for t in os.environ["E5_TASKS"].split(",") if t.strip())
    _unknown = [t for t in _want if t not in _ALL_TASKS]
    if _unknown or not _want:
        raise ValueError(f"E5_TASKS={os.environ['E5_TASKS']!r} names unknown task(s) {_unknown} — "
                         f"pick a comma-separated subset of {_ALL_TASKS}")
    TASKS = _want

# Languages each HF dataset provides, in our canonical codes (XWinograd's `jp` -> `ja`).
_BENCHMARK_LANGS: dict[str, tuple[str, ...]] = {
    "mgsm": ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"),
    "mmlu": ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "zh"),   # Global-MMLU: no `th`
    "xwinograd": ("en", "fr", "ja", "pt", "ru", "zh"),                       # `jp` normalized to `ja`
}

_HF_DATASET: dict[str, str] = {
    "mgsm": "juletxara/mgsm",
    "mmlu": "CohereLabs/Global-MMLU",
    "xwinograd": "Muennighoff/xwinograd",
}

# Map a canonical language code to that dataset's HF config name (only XWinograd differs: `ja`->`jp`).
_XWINOGRAD_CONFIG = {"ja": "jp"}


@dataclass(frozen=True)
class BenchItem:
    """One benchmark example, benchmark-agnostic (docs/modules/benchmarks.md §2)."""
    task: str                              # "mgsm" | "mmlu" | "xwinograd"
    lang: str                              # our canonical code (jp -> ja normalized)
    prompt: str                            # fully built prompt / log-likelihood context
    gold: str                              # numeric string | "A".."D" | "1"/"2"
    choices: tuple[str, ...] = ()          # MC/cloze continuations; () for generative (MGSM)
    meta: dict = field(default_factory=dict)  # id, subject, raw fields, few-shot used


def benchmark_languages(task: str) -> tuple[str, ...]:
    """Languages the HF dataset provides for `task`, in our canonical codes."""
    if task not in _BENCHMARK_LANGS:
        raise ValueError(f"unknown task {task!r} (expected one of {TASKS})")
    return _BENCHMARK_LANGS[task]


def eval_languages(task: str, cfg: Config) -> tuple[str, ...]:
    """`benchmark_languages(task) ∩ cfg.langs` (the E5 §3 eval-language policy).

    Ordered by `cfg.langs` so downstream loops are stable. MGSM -> 11, M-MMLU -> 10, XWinograd -> 5.
    """
    provided = set(benchmark_languages(task))
    return tuple(l for l in cfg.langs if l in provided)


def _hf_config(task: str, lang: str) -> str:
    """Dataset config name for a canonical language (XWinograd stores Japanese under `jp`)."""
    if task == "xwinograd":
        return _XWINOGRAD_CONFIG.get(lang, lang)
    return lang


def _load_hf_split(task: str, lang: str, split: str):
    """Thin wrapper over `datasets.load_dataset`, cached under `data/benchmarks/` (git-ignored)."""
    from datasets import load_dataset

    from extract import _configure_hf_http

    _configure_hf_http()
    BENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        _HF_DATASET[task], _hf_config(task, lang), split=split, cache_dir=str(BENCH_CACHE_DIR)
    )


def _subsample(items: list[BenchItem], n: int | None, seed: int) -> list[BenchItem]:
    """Seeded size-`n` subsample (D6). Keeps original order; no-op when `n` is None or >= len."""
    if n is None or n >= len(items):
        return items
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(items), size=n, replace=False))
    return [items[i] for i in pick]


def _load_mgsm(lang: str, split: str, *, n_shot: int) -> list[BenchItem]:
    exemplars: list[dict] = []
    if n_shot > 0:
        train = _load_hf_split("mgsm", lang, "train")
        exemplars = [{"question": r["question"], "answer": r.get("answer") or ""}
                     for r in list(train)[:n_shot]]
    ds = _load_hf_split("mgsm", lang, split)
    items: list[BenchItem] = []
    for i, r in enumerate(ds):
        item = {"question": r["question"], "lang": lang}
        prompt = build_mgsm_prompt(item, exemplars, n_shot=n_shot, instr_lang=lang)
        items.append(BenchItem(
            task="mgsm", lang=lang, prompt=prompt,
            gold=str(r["answer_number"]), choices=(),
            meta={"id": f"mgsm-{lang}-{i}", "n_shot": n_shot, "instr_lang": lang,
                  "equation_solution": r.get("equation_solution")},
        ))
    return items


def _load_mmlu(lang: str, split: str) -> list[BenchItem]:
    ds = _load_hf_split("mmlu", lang, split)
    letters = ("A", "B", "C", "D")
    items: list[BenchItem] = []
    for i, r in enumerate(ds):
        item = {
            "question": r["question"], "lang": lang,
            "option_a": r["option_a"], "option_b": r["option_b"],
            "option_c": r["option_c"], "option_d": r["option_d"],
        }
        prompt = build_mmlu_prompt(item, instr_lang=lang)
        items.append(BenchItem(
            task="mmlu", lang=lang, prompt=prompt,
            gold=str(r["answer"]).strip().upper(), choices=letters,
            meta={"id": f"mmlu-{lang}-{i}", "subject": r.get("subject"), "instr_lang": lang},
        ))
    return items


def _load_xwinograd(lang: str, split: str) -> list[BenchItem]:
    ds = _load_hf_split("xwinograd", lang, split)
    items: list[BenchItem] = []
    for i, r in enumerate(ds):
        cont1, cont2 = build_xwinograd_continuations(
            {"sentence": r["sentence"], "option1": r["option1"], "option2": r["option2"]}
        )
        items.append(BenchItem(
            task="xwinograd", lang=lang, prompt="",           # pure continuation scoring, no context
            gold=str(r["answer"]).strip(), choices=(cont1, cont2),
            meta={"id": f"xwinograd-{lang}-{i}", "sentence": r["sentence"],
                  "option1": r["option1"], "option2": r["option2"]},
        ))
    return items


def load_benchmark(
    task: str, lang: str, split: str = "test", *,
    cfg: Config, n: int | None = None, seed: int = 0, n_shot: int = 0,
) -> list[BenchItem]:
    """Load one `(task, lang)` split into `BenchItem`s.

    Maps native dataset fields into the unified schema, builds prompts (MGSM pulls `n_shot`
    localized exemplars from the `train` split), normalizes XWinograd `jp`->`ja` on load, and
    optionally takes a seeded size-`n` subsample (D6). `cfg` is accepted for API symmetry /
    forward compatibility (HF endpoints are dataset-fixed, not `cfg`-derived).
    """
    if lang not in benchmark_languages(task):
        raise ValueError(f"{task!r} does not provide language {lang!r} "
                         f"(available: {benchmark_languages(task)})")
    if task == "mgsm":
        items = _load_mgsm(lang, split, n_shot=n_shot)
    elif task == "mmlu":
        items = _load_mmlu(lang, split)
    elif task == "xwinograd":
        items = _load_xwinograd(lang, split)
    else:
        raise ValueError(f"unknown task {task!r} (expected one of {TASKS})")
    return _subsample(items, n, seed)


def split_dev_test(
    items: list[BenchItem], dev_frac: float = 0.5, seed: int = 0
) -> tuple[list[BenchItem], list[BenchItem]]:
    """Seeded dev/test split on item id (data-role discipline, E5 §2).

    Splitting on the stable `meta["id"]` (not list position) keeps the same item in the same role
    regardless of input ordering. Returns `(dev, test)`, each in the input order.
    """
    ids = [str(it.meta.get("id", i)) for i, it in enumerate(items)]
    # permute the id-sorted positions so the assignment depends on id, not on input order
    order = sorted(range(len(items)), key=lambda i: ids[i])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(order))
    cut = int(round(dev_frac * len(order)))
    dev_positions = {order[p] for p in perm[:cut].tolist()}
    dev = [it for i, it in enumerate(items) if i in dev_positions]
    test = [it for i, it in enumerate(items) if i not in dev_positions]
    return dev, test
