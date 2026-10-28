"""Harness tests: log-likelihood scoring on a stubbed model, aggregation shape, real-model smoke."""
import types

import pandas as pd
import pytest
import torch

from benchmarks import BenchItem
from evaluate import (
    aggregate,
    low_resource_summary,
    paired_delta,
    run_eval,
    score_continuations,
)


class StubTokenizer:
    """Char-level tokenizer: token id = ord(char), so tests are fully deterministic."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, **kw):
        return {"input_ids": [ord(c) for c in text]}


class StubModel:
    """Returns position-independent logits that strongly prefer a single target token id."""

    def __init__(self, target_id: int, vocab: int = 512):
        self.target_id = target_id
        self.vocab = vocab
        self.config = types.SimpleNamespace(_name_or_path="stub/StubModel")

    def __call__(self, input_ids):
        T = input_ids.shape[1]
        logits = torch.full((1, T, self.vocab), -10.0)
        logits[..., self.target_id] = 10.0
        return types.SimpleNamespace(logits=logits)


def test_score_continuations_argmax_picks_favored_token():
    tok = StubTokenizer()
    model = StubModel(target_id=ord("2"))
    scores = score_continuations(model, tok, "Q", ["1", "2"], device="cpu")
    assert scores[1] > scores[0]                      # the continuation containing '2' wins


def test_score_continuations_token_norm_changes_scale_not_favored_sign():
    tok = StubTokenizer()
    model = StubModel(target_id=ord("2"))
    raw = score_continuations(model, tok, "Q", ["22", "1"], device="cpu")
    normed = score_continuations(model, tok, "Q", ["22", "1"], device="cpu", token_norm=True)
    assert raw[0] > raw[1]                            # "22" has two favored tokens -> higher sum
    # token-normalized both are per-token; "22" per-token ~ favored, "1" per-token ~ unfavored
    assert normed[0] > normed[1]


def test_aggregate_shape_and_columns():
    df = pd.DataFrame([
        {"model": "M", "task": "mgsm", "lang": "en", "correct": True, "cond": "clean"},
        {"model": "M", "task": "mgsm", "lang": "en", "correct": False, "cond": "clean"},
        {"model": "M", "task": "mmlu", "lang": "de", "correct": True, "cond": "clean"},
    ])
    agg = aggregate(df, n_boot=50, seed=0)
    assert set(agg.columns) >= {"model", "task", "lang", "n", "acc", "acc_lo", "acc_hi", "cond"}
    assert len(agg) == 2                              # (mgsm,en) and (mmlu,de)
    row = agg[(agg["task"] == "mgsm") & (agg["lang"] == "en")].iloc[0]
    assert row["n"] == 2 and abs(row["acc"] - 0.5) < 1e-9


def _paired_frame():
    """Two conditions on identical items: 'sw' (low-resource) gains more than 'en' under cond_b."""
    rows = []
    # en (high-resource): cond_a correct [1, 0], cond_b [1, 1]  -> per-item delta (0, 1), mean 0.5
    for id_, (a, b) in {"e0": (1, 1), "e1": (0, 1)}.items():
        rows.append({"model": "M", "task": "mgsm", "lang": "en", "id": id_, "correct": bool(a), "cond": "alg1_tuned"})
        rows.append({"model": "M", "task": "mgsm", "lang": "en", "id": id_, "correct": bool(b), "cond": "diffpca_tuned"})
    # sw (low-resource): cond_a [0, 0], cond_b [1, 1] -> per-item delta (1, 1), mean 1.0
    for id_, (a, b) in {"s0": (0, 1), "s1": (0, 1)}.items():
        rows.append({"model": "M", "task": "mgsm", "lang": "sw", "id": id_, "correct": bool(a), "cond": "alg1_tuned"})
        rows.append({"model": "M", "task": "mgsm", "lang": "sw", "id": id_, "correct": bool(b), "cond": "diffpca_tuned"})
    return pd.DataFrame(rows)


def test_paired_delta_sign_magnitude_and_low_resource_flag():
    df = _paired_frame()
    pd_df = paired_delta(df, "diffpca_tuned", "alg1_tuned", n_boot=50, seed=0)
    assert set(pd_df.columns) >= {
        "model", "task", "lang", "n", "acc_a", "acc_b", "delta", "delta_lo", "delta_hi", "low_resource",
    }
    en = pd_df[pd_df["lang"] == "en"].iloc[0]
    sw = pd_df[pd_df["lang"] == "sw"].iloc[0]
    assert abs(en["delta"] - 0.5) < 1e-9 and abs(en["acc_a"] - 0.5) < 1e-9 and abs(en["acc_b"] - 1.0) < 1e-9
    assert abs(sw["delta"] - 1.0) < 1e-9
    assert not en["low_resource"] and sw["low_resource"]
    assert en["n"] == 2 and sw["n"] == 2


def test_paired_delta_enforces_id_alignment():
    df = _paired_frame()
    # drop one cond_b row so 'en' no longer pairs 1:1
    bad = df.drop(df[(df["lang"] == "en") & (df["id"] == "e1") & (df["cond"] == "diffpca_tuned")].index)
    with pytest.raises(ValueError):
        paired_delta(bad, "diffpca_tuned", "alg1_tuned", n_boot=10)


def test_low_resource_summary_groups_and_pools():
    df = _paired_frame()
    pd_df = paired_delta(df, "diffpca_tuned", "alg1_tuned", n_boot=50, seed=0)
    # fallback path (bootstrap over per-language deltas)
    summ = low_resource_summary(pd_df, n_boot=50, seed=0)
    assert set(summ.columns) >= {"low_resource", "n_langs", "delta", "delta_lo", "delta_hi"}
    assert len(summ) == 2
    lo_row = summ[summ["low_resource"]].iloc[0]
    hi_row = summ[~summ["low_resource"]].iloc[0]
    assert abs(lo_row["delta"] - 1.0) < 1e-9        # sw pools to +1.0
    assert lo_row["delta"] > hi_row["delta"]         # H3: low-resource edge is larger
    assert lo_row["n_langs"] == 1 and hi_row["n_langs"] == 1


def test_low_resource_summary_pooled_items_path():
    df = _paired_frame()
    pd_df = paired_delta(df, "diffpca_tuned", "alg1_tuned", n_boot=50, seed=0)
    # pooled-items path: pass the per-item frame + cond labels
    summ = low_resource_summary(
        pd_df, df, cond_b="diffpca_tuned", cond_a="alg1_tuned", n_boot=50, seed=0
    )
    lo_row = summ[summ["low_resource"]].iloc[0]
    assert abs(lo_row["delta"] - 1.0) < 1e-9          # pooled sw items still +1.0


def test_run_eval_rejects_unknown_task():
    it = BenchItem(task="bogus", lang="en", prompt="p", gold="1")
    with pytest.raises(ValueError):
        run_eval(StubModel(0), StubTokenizer(), [it], device="cpu")


@pytest.mark.slow
def test_run_eval_real_model_smoke():
    from config import Config, get_device
    from extract import load_model_and_tokenizer

    cfg = Config()
    device = get_device()
    model, tok = load_model_and_tokenizer(cfg, device)
    items = [
        BenchItem("mmlu", "en", "The capital of France is\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\nAnswer:",
                  gold="B", choices=("A", "B", "C", "D"), meta={"id": "smoke-mmlu-0"}),
        BenchItem("xwinograd", "en", "", gold="1",
                  choices=("The cat ran because the cat was scared.",
                           "The cat ran because the dog was scared."),
                  meta={"id": "smoke-xwino-0"}),
    ]
    df = run_eval(model, tok, items, device=device, model_name=cfg.model_name)
    assert len(df) == 2
    assert set(df.columns) >= {"model", "task", "lang", "pred", "gold", "correct", "cond"}


# --- generative path: greedy vs reasoning sampling, </think> strip, two fidelities ---------------

class _Enc(dict):
    def to(self, device):
        return self


class GenTokenizer:
    """Char-level generative tokenizer: records `enable_thinking`; canned char<->id decode."""
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __init__(self):
        self.seen_enable_thinking = []

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kw):
        self.seen_enable_thinking.append(kw.get("enable_thinking", "ABSENT"))
        return msg[0]["content"]

    def __call__(self, text, return_tensors=None, padding=False, **kw):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        ids = [[ord(c) for c in t] for t in texts]
        if return_tensors == "pt":
            m = max((len(x) for x in ids), default=1)
            padded = [[self.pad_token_id] * (m - len(x)) + x for x in ids]   # left-pad
            return _Enc(input_ids=torch.tensor(padded, dtype=torch.long))
        return {"input_ids": ids[0] if single else ids}

    def batch_decode(self, tensor, skip_special_tokens=True):
        return ["".join(chr(int(i)) for i in row if int(i) != self.pad_token_id)
                for row in tensor.tolist()]


class GenModel:
    """Appends a canned continuation to the prompt; records generate() kwargs for assertions."""
    def __init__(self, canned="Answer: 42", name="stub/StubModel"):
        self.canned = canned
        self.config = types.SimpleNamespace(_name_or_path=name)
        self.generation_config = types.SimpleNamespace(eos_token_id=1)
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.calls.append(kw)
        cont = torch.tensor([[ord(c) for c in self.canned]] * input_ids.shape[0], dtype=torch.long)
        return torch.cat([input_ids, cont], dim=1)


def _mgsm(gold="42", lang="en", id_="m0"):
    return BenchItem("mgsm", lang, "solve: 6*7", gold=gold, meta={"id": id_})


def test_generative_default_is_greedy_thinking_off():
    tok, model = GenTokenizer(), GenModel(canned="Answer: 42")
    df = run_eval(model, tok, [_mgsm()], device="cpu")
    call = model.calls[-1]
    assert call["do_sample"] is False and "temperature" not in call
    assert "eos_token_id" not in call                        # default spec => no eos override
    assert tok.seen_enable_thinking[-1] is False
    assert df.iloc[0]["pred"] == "42" and bool(df.iloc[0]["correct"])


def test_generative_reasoning_samples_and_thinks_and_strips_trace():
    tok = GenTokenizer()
    model = GenModel(canned="<think>99</think>\nAnswer: 42", name="Qwen/Qwen3-8B")
    df = run_eval(model, tok, [_mgsm()], device="cpu", model_name="Qwen/Qwen3-8B")
    call = model.calls[-1]
    assert call["do_sample"] is True
    assert call["temperature"] == 0.6 and call["top_p"] == 0.95 and call["top_k"] == 20
    assert call["max_new_tokens"] == 16384
    assert tok.seen_enable_thinking[-1] is True
    assert df.iloc[0]["pred"] == "42"                        # trace number 99 stripped, answer scored
    out = str(df.iloc[0]["output"])                          # `output` holds the STRIPPED answer...
    assert "99" not in out and "<think>" not in out          # ...load-bearing: strip actually ran
    assert df.iloc[0]["reasoning_lang"] != ""                # reasoning-fidelity signal available


def test_generative_forced_think_truncated_scores_no_answer():
    # R1-distill / GLM-Z1: the opening <think> is emitted PROMPT-side, so a trace truncated at the
    # cap decodes to neither tag. It must be treated as a truncated trace (no answer), NOT scored as
    # the last mid-reasoning number. This is the rank-1 review finding.
    tok = GenTokenizer()
    model = GenModel(canned="let me think, maybe 7, then 13",
                     name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    df = run_eval(model, tok, [_mgsm(gold="42")], device="cpu",
                  model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    assert df.iloc[0]["pred"] == ""                          # truncated trace -> no scorable answer
    assert not bool(df.iloc[0]["correct"])                   # '13' was NOT scored against gold 42
    assert df.iloc[0]["reasoning_lang"] != ""                # the whole output is (truncated) reasoning


def test_generative_explicit_override_beats_spec():
    from models import GenConfig
    tok = GenTokenizer()
    model = GenModel(canned="Answer: 1", name="Qwen/Qwen3-8B")
    run_eval(model, tok, [_mgsm(gold="1")], device="cpu", model_name="Qwen/Qwen3-8B",
             enable_thinking=False, gen=GenConfig())          # force greedy / thinking-off
    call = model.calls[-1]
    assert call["do_sample"] is False and "temperature" not in call
    assert tok.seen_enable_thinking[-1] is False


def test_aggregate_emits_response_and_reasoning_fidelity():
    df = pd.DataFrame([
        {"model": "M", "task": "mgsm", "lang": "en", "correct": True,
         "cond": "clean", "output_lang": "en", "reasoning_lang": "en"},
        {"model": "M", "task": "mgsm", "lang": "en", "correct": False,
         "cond": "clean", "output_lang": "zh", "reasoning_lang": "en"},
    ])
    agg = aggregate(df, n_boot=20, seed=0)
    row = agg.iloc[0]
    assert abs(row["response_fidelity"] - 0.5) < 1e-9        # 1 of 2 answers in target language
    assert abs(row["reasoning_fidelity"] - 1.0) < 1e-9       # both traces in target language


def test_generative_logs_truncation_instrument():
    tok, model = GenTokenizer(), GenModel(canned="Answer: 42")
    df = run_eval(model, tok, [_mgsm()], device="cpu")                 # cap 512, short output
    assert bool(df.iloc[0]["truncated"]) is False and df.iloc[0]["n_new_tokens"] > 0
    assert int(df.iloc[0]["max_new_tokens"]) == 512
    df2 = run_eval(model, tok, [_mgsm()], device="cpu", max_new_tokens=5)   # force a cap-hit
    assert bool(df2.iloc[0]["truncated"]) is True
    agg = aggregate(df2, n_boot=10, seed=0)
    assert abs(agg.iloc[0]["truncation_rate"] - 1.0) < 1e-9


def test_iut_cross_verdict_rules():
    from evaluate import iut_cross_verdict
    assert iut_cross_verdict((0.02, 0.06), (0.01, 0.05)) == "compounds"        # both CIs > 0
    assert iut_cross_verdict((-0.05, -0.01), (0.01, 0.05)) == "interferes"     # one CI < 0
    assert iut_cross_verdict((-0.005, 0.005), (-0.004, 0.004), mde=0.01) == "redundant"
    assert iut_cross_verdict((-0.02, 0.06), (0.01, 0.05)) == "indeterminate"   # one CI spans 0


def test_classify_crosses_from_paired_frame():
    from evaluate import classify_crosses
    rows = [
        {"model": "M", "task": "mgsm", "lang": "en", "contrast": "diffpca_oblique_tuned - diffpca_tuned",
         "delta": 0.04, "delta_lo": 0.02, "delta_hi": 0.06},
        {"model": "M", "task": "mgsm", "lang": "en", "contrast": "diffpca_oblique_tuned - leace_tuned",
         "delta": 0.03, "delta_lo": 0.01, "delta_hi": 0.05},
    ]
    out = classify_crosses(pd.DataFrame(rows), {"diffpca_oblique": ("diffpca", "leace")})
    assert out.iloc[0]["verdict"] == "compounds" and out.iloc[0]["cross"] == "diffpca_oblique"


def test_holm_adjust_hand_computed_stepdown():
    from evaluate import holm_adjust
    # family {a:.01, c:.03, b:.04}, alpha=.05, m=3. Sorted ascending: a, c, b.
    out = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert list(out["key"]) == ["a", "c", "b"]                    # returned ascending by p
    got = {r["key"]: (round(float(r["p_holm"]), 4), bool(r["reject"])) for _, r in out.iterrows()}
    assert got["a"] == (0.03, True)      # .01 <= .05/3=.0167 -> reject; adj = 3*.01 = .03
    assert got["c"] == (0.06, False)     # .03 > .05/2=.025 -> stop; adj = max(.03, 2*.03) = .06
    assert got["b"] == (0.06, False)     # step-down closed; adj = max(.06, 1*.04) = .06


def test_holm_adjust_all_reject_and_empty():
    from evaluate import holm_adjust
    out = holm_adjust([("x", 0.001), ("y", 0.002)], alpha=0.05)   # both well under thresholds
    assert bool(out.set_index("key").loc["x", "reject"]) and bool(out.set_index("key").loc["y", "reject"])
    empty = holm_adjust({})
    assert list(empty.columns) == ["key", "p", "p_holm", "reject"] and len(empty) == 0
