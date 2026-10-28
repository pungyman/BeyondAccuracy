"""Unit tests for the stratified problem sampler (no model; data-file tests self-skip)."""
import collections

import numpy as np
import pytest

from config import DATA_DIR, Config
from corpus import (
    _load_records,
    load_parallel,
    select_problem_indices,
    stratified_indices,
)

# synthetic, skewed strata: 5 levels x 7 types, uneven sizes (mirrors the MATH skew)
RNG = np.random.default_rng(123)
LEVELS = [f"Level {i}" for i in range(1, 6)]
TYPES = ["Algebra", "Geometry", "Number Theory", "Precalculus",
         "Prealgebra", "Counting", "Intermediate"]
_SYNTH_LABELS = []
for li, lv in enumerate(LEVELS):
    for ti, tp in enumerate(TYPES):
        size = 20 + 13 * li + 7 * ti          # deterministic, uneven per-stratum sizes
        _SYNTH_LABELS += [(lv, tp)] * size
RNG.shuffle(_SYNTH_LABELS)
_SYNTH_LABELS = [tuple(x) for x in _SYNTH_LABELS]
N_SYNTH = len(_SYNTH_LABELS)

DATA_OK = (DATA_DIR / "en.json").exists()


def test_count_unique_sorted_in_range():
    idx = stratified_indices(_SYNTH_LABELS, n=300, seed=0)
    assert len(idx) == 300
    assert len(set(idx.tolist())) == 300                 # no duplicates
    assert idx.min() >= 0 and idx.max() < N_SYNTH
    assert np.array_equal(idx, np.sort(idx))             # sorted


def test_determinism_same_seed():
    a = stratified_indices(_SYNTH_LABELS, n=250, seed=7)
    b = stratified_indices(_SYNTH_LABELS, n=250, seed=7)
    assert np.array_equal(a, b)


def test_different_seed_changes_selection():
    a = stratified_indices(_SYNTH_LABELS, n=250, seed=1)
    b = stratified_indices(_SYNTH_LABELS, n=250, seed=2)
    assert not np.array_equal(a, b)


def test_proportional_shares_within_one():
    n = 400
    idx = stratified_indices(_SYNTH_LABELS, n=n, seed=0)
    sizes = collections.Counter(_SYNTH_LABELS)
    got = collections.Counter(_SYNTH_LABELS[i] for i in idx)
    for stratum, s_k in sizes.items():
        expected = n * s_k / N_SYNTH
        assert abs(got[stratum] - expected) <= 1.0       # largest-remainder => within one slot


def test_full_and_empty_edges():
    assert np.array_equal(stratified_indices(_SYNTH_LABELS, n=N_SYNTH, seed=0),
                          np.arange(N_SYNTH))
    assert stratified_indices(_SYNTH_LABELS, n=0, seed=0).size == 0


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        stratified_indices(_SYNTH_LABELS, n=N_SYNTH + 1, seed=0)


def test_degenerate_tiny_stratum_and_small_n():
    labels = [("Level ?", "Geometry"), ("Level ?", "Geometry")] + [("Level 1", "Algebra")] * 50
    idx = stratified_indices(labels, n=8, seed=0)         # must not crash on the size-2 stratum
    assert len(idx) == 8
    assert len(set(idx.tolist())) == 8


def test_select_first_mode_is_legacy_arange():
    records = [{"level": "L", "type": "T", "outcome": str(i)} for i in range(20)]
    idx, labels = select_problem_indices(records, n=5, sample="first")
    assert np.array_equal(idx, np.arange(5))
    assert labels[0] == {"level": "L", "type": "T", "outcome": "0"}


def test_select_unknown_mode_raises():
    records = [{"level": "L", "type": "T"} for _ in range(5)]
    with pytest.raises(ValueError):
        select_problem_indices(records, n=2, sample="bogus")


def test_select_labels_aligned_to_indices():
    records = [{"level": LEVELS[i % 5], "type": TYPES[i % 7], "outcome": str(i)}
               for i in range(200)]
    idx, labels = select_problem_indices(records, n=40, sample="stratified", seed=0)
    assert len(labels) == len(idx) == 40
    for pos, i in enumerate(idx):
        assert labels[pos]["outcome"] == str(i)          # label row matches its source record


@pytest.mark.skipif(not DATA_OK, reason="parallel corpus not present locally")
def test_real_corpus_parallel_and_stratified():
    cfg = Config(n_samples=120, sample="stratified", seed=0)
    langs = ["en", "te", "zh"]
    records = _load_records("en", cfg)
    idx, _ = select_problem_indices(records, cfg.n_samples, sample=cfg.sample,
                                    seed=cfg.seed, strata_keys=cfg.strata_keys)
    prompts = load_parallel(langs, cfg.n_samples, cfg.prompt_key, cfg, indices=idx)
    assert {len(v) for v in prompts.values()} == {cfg.n_samples}   # parallel across langs
    # stratified selection should not be the first-n slice on this skewed corpus
    assert not np.array_equal(idx, np.arange(cfg.n_samples))


# --- build_chat_inputs enable_thinking threading -------------------------------------------------

class _RecordingTokenizer:
    """Records the `enable_thinking` value the template was called with."""
    def __init__(self):
        self.seen = []

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kw):
        self.seen.append(kw.get("enable_thinking", "ABSENT"))
        return msg[0]["content"]


class _NoThinkingTokenizer:
    """A template that does NOT accept the kwarg (raises TypeError) -> fallback path."""
    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True):
        return "FALLBACK:" + msg[0]["content"]


def test_build_chat_inputs_defaults_thinking_off_and_threads_true():
    from corpus import build_chat_inputs
    tok = _RecordingTokenizer()
    build_chat_inputs(tok, ["p"])
    assert tok.seen[-1] is False                          # extraction / non-reasoning default
    build_chat_inputs(tok, ["p"], enable_thinking=True)
    assert tok.seen[-1] is True                           # reasoning eval


def test_build_chat_inputs_typeerror_fallback():
    from corpus import build_chat_inputs
    out = build_chat_inputs(_NoThinkingTokenizer(), ["hi"])   # rejects enable_thinking -> falls back
    assert out == ["FALLBACK:hi"]
