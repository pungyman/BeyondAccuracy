"""Schema / coverage / prompt-builder tests for the E5 loaders (offline; real loads self-skip)."""
import os

import pytest

from config import Config
from benchmarks import (
    BenchItem,
    _hf_config,
    benchmark_languages,
    eval_languages,
    load_benchmark,
    split_dev_test,
)
from prompts import build_mgsm_prompt, build_mmlu_prompt, build_xwinograd_continuations

RUN_DATASET = os.environ.get("LRD_RUN_DATASET_TESTS") == "1"


def _items(task, lang, n):
    return [BenchItem(task, lang, f"p{i}", str(i), (), {"id": f"{task}-{lang}-{i}"}) for i in range(n)]


def test_eval_languages_are_the_spec_intersections():
    cfg = Config()
    assert eval_languages("mgsm", cfg) == ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")
    assert eval_languages("mmlu", cfg) == ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "zh")
    assert eval_languages("xwinograd", cfg) == ("en", "fr", "ja", "ru", "zh")


def test_eval_languages_respects_cfg_subset_and_order():
    cfg = Config(langs=("zh", "en", "th"))
    assert eval_languages("mgsm", cfg) == ("zh", "en", "th")     # ordered by cfg.langs
    assert eval_languages("mmlu", cfg) == ("zh", "en")           # th absent from Global-MMLU


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        benchmark_languages("bogus")


def test_xwinograd_jp_normalized_to_ja():
    # exposed as `ja` in our canonical language set, stored as `jp` in the HF config
    assert "ja" in benchmark_languages("xwinograd")
    assert "jp" not in benchmark_languages("xwinograd")
    assert _hf_config("xwinograd", "ja") == "jp"
    assert _hf_config("xwinograd", "en") == "en"
    assert _hf_config("mgsm", "ja") == "ja"                      # only XWinograd remaps


def test_bench_item_defaults():
    it = BenchItem(task="mgsm", lang="en", prompt="p", gold="1")
    assert it.choices == () and it.meta == {}


def test_xwinograd_continuation_building():
    c1, c2 = build_xwinograd_continuations(
        {"sentence": "The trophy didn't fit in the case because _ was too big.",
         "option1": "the trophy", "option2": "the case"}
    )
    assert c1 == "The trophy didn't fit in the case because the trophy was too big."
    assert c2 == "The trophy didn't fit in the case because the case was too big."
    assert "_" not in c1 and "_" not in c2


def test_mgsm_prompt_uses_native_keyword_and_exemplars():
    p = build_mgsm_prompt({"question": "How many?", "lang": "de"},
                          [{"question": "Q1", "answer": "Loesung. Antwort: 5"}],
                          n_shot=1, instr_lang="de")
    assert "Q1" in p and "How many?" in p
    assert p.rstrip().endswith("Antwort:")                       # localized cue, no \\boxed{}
    assert "\\boxed" not in p


def test_mmlu_prompt_has_abcd_block_and_localized_cue():
    p = build_mmlu_prompt({"question": "2+2?", "lang": "fr", "option_a": "3",
                           "option_b": "4", "option_c": "5", "option_d": "6"})
    for line in ("A. 3", "B. 4", "C. 5", "D. 6"):
        assert line in p
    assert p.rstrip().endswith("R\u00e9ponse:")                  # French "Réponse:"


def test_split_dev_test_deterministic_and_disjoint():
    items = _items("mgsm", "en", 20)
    dev, test = split_dev_test(items, dev_frac=0.5, seed=0)
    assert len(dev) == 10 and len(test) == 10
    dev_ids = {it.meta["id"] for it in dev}
    test_ids = {it.meta["id"] for it in test}
    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == {it.meta["id"] for it in items}
    # same seed -> same split
    dev2, _ = split_dev_test(items, dev_frac=0.5, seed=0)
    assert {it.meta["id"] for it in dev2} == dev_ids


def test_split_dev_test_is_order_invariant():
    items = _items("mmlu", "en", 16)
    dev_a, _ = split_dev_test(items, seed=3)
    dev_b, _ = split_dev_test(list(reversed(items)), seed=3)
    assert {it.meta["id"] for it in dev_a} == {it.meta["id"] for it in dev_b}


@pytest.mark.skipif(not RUN_DATASET, reason="set LRD_RUN_DATASET_TESTS=1 to hit the HF hub")
def test_real_load_mgsm_schema():
    cfg = Config()
    items = load_benchmark("mgsm", "en", "test", cfg=cfg, n=4, seed=0, n_shot=2)
    assert len(items) == 4
    for it in items:
        assert it.task == "mgsm" and it.lang == "en" and it.choices == ()
        assert it.gold and it.prompt.rstrip().endswith("Answer:")
