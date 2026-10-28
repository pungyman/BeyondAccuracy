"""Unit tests for the E5 scorers + the previously-untested numeric MGSM path (no model, no data).

Covers the native-route MGSM extraction on non-Latin numerals (`bn te th`), MC letter extraction,
2-way cloze argmax, and the existing `extract_pred`/`norm`/`is_correct` numeric path.
"""
from scoring import (
    _ascii_digits,
    extract_choice,
    extract_mgsm_answer,
    extract_pred,
    is_correct,
    is_correct_mc,
    norm,
    score_cloze,
)


def test_ascii_digits_bn_te_th():
    assert _ascii_digits("\u09e7\u09e8\u09e9") == "123"          # Bengali 1 2 3
    assert _ascii_digits("\u0c66\u0c67") == "01"                  # Telugu 0 1
    assert _ascii_digits("\u0e57") == "7"                          # Thai 7
    assert _ascii_digits("mix \u09ea2") == "mix 42"                # Bengali 4 + ASCII 2


def test_extract_mgsm_prefers_after_localized_keyword():
    # a distractor number appears before the answer keyword; the keyword-anchored one wins
    assert extract_mgsm_answer("First 5 apples, then 6. Answer: 11", "en") == "11"


def test_extract_mgsm_native_numerals():
    assert extract_mgsm_answer("\u0989\u09a4\u09cd\u09a4\u09b0: \u09ea\u09e8", "bn") == "42"   # উত্তর: ৪২
    assert extract_mgsm_answer("\u0938\u092e\u093e\u0927\u093e\u0928\u0902: \u0c67\u0c66\u0c66", "te") == "100"
    assert extract_mgsm_answer("\u0e04\u0e33\u0e15\u0e2d\u0e1a: \u0e57", "th") == "7"


def test_extract_mgsm_fallback_last_number_without_keyword():
    assert extract_mgsm_answer("the total comes to 24 in the end", "de") == "24"
    assert extract_mgsm_answer("no digits here", "en") == ""


def test_extract_mgsm_answer_is_correct_across_scripts():
    assert is_correct(extract_mgsm_answer("\u09ea\u09e8 \u0989\u09a4\u09cd\u09a4\u09b0: \u09ea\u09e8", "bn"), "42")


def test_extract_choice_cue_and_standalone():
    assert extract_choice("blah blah \u7b54\u6848: C") == "C"          # localized "答案: C"
    assert extract_choice("I'll go with (D) here") == "D"
    assert extract_choice("The answer is B.") == "B"
    assert extract_choice("no letters in range") == ""


def test_extract_choice_takes_last_when_multiple():
    assert extract_choice("maybe A, actually C") == "C"


def test_is_correct_mc():
    assert is_correct_mc("a", "A") is True
    assert is_correct_mc(" C ", "C") is True
    assert is_correct_mc("B", "A") is False
    assert is_correct_mc("", "A") is False


def test_score_cloze_argmax():
    assert score_cloze((0.1, 0.9)) == "2"
    assert score_cloze((0.9, 0.1)) == "1"
    assert score_cloze((0.5, 0.5)) == "1"          # ties -> option 1


def test_numeric_path_extract_pred_and_norm():
    assert extract_pred("...\\boxed{42}") == "42"
    assert extract_pred("the answer is 3/4") == "3/4"
    assert norm("\\boxed{ 1,000 }") == "1000"
    assert is_correct("1/2", "0.5") is True
    assert is_correct("", "5") is False


# --- reasoning-trace stripping (thinking-model answer parse) --------------------------------------

def test_strip_reasoning_drops_trace_keeps_answer():
    from scoring import strip_reasoning
    assert strip_reasoning("<think>work 5</think>\nAnswer: 12") == "\nAnswer: 12"


def test_strip_reasoning_last_close_wins():
    from scoring import strip_reasoning
    assert strip_reasoning("<think>a</think>x<think>b</think>Y") == "Y"


def test_strip_reasoning_no_tags_is_noop():
    from scoring import strip_reasoning
    assert strip_reasoning("Answer: 7") == "Answer: 7"


def test_strip_reasoning_truncated_trace_returns_empty():
    from scoring import strip_reasoning
    # opened but never closed (hit the token cap mid-thought) -> no answer emitted
    assert strip_reasoning("<think>reasoning 9 with no close") == ""


def test_strip_reasoning_expect_trace_truncated_returns_empty():
    from scoring import strip_reasoning
    # forced-<think> model (GLM-Z1/R1): the opening <think> is prompt-side, so a truncated
    # continuation carries NEITHER tag. expect_trace=True must treat it as a truncated trace -> "".
    assert strip_reasoning("so the answer is probably 7, wait recompute", expect_trace=True) == ""
    # the same text with expect_trace=False (a non-reasoning answer) passes through unchanged
    assert strip_reasoning("so the answer is probably 7", expect_trace=False).endswith("7")


def test_strip_then_extract_no_answer_on_truncated_trace():
    from scoring import extract_mgsm_answer, strip_reasoning
    # load-bearing: without the expect_trace guard, extract would grab '13' from mid-reasoning;
    # with it, the truncated trace yields no scorable answer.
    assert extract_mgsm_answer(strip_reasoning("hmm 7 then 13", expect_trace=True), "en") == ""
    assert extract_mgsm_answer(strip_reasoning("hmm 7 then 13", expect_trace=False), "en") == "13"


# --- hardened output-language detection (fidelity axis) -------------------------------------------

def test_detlang_resolves_nonlatin_by_script():
    from scoring import detlang
    assert detlang("这是一个数学题，答案是 42") == "zh"        # Han (no kana) -> zh
    assert detlang("答えは 42 です。にほんご") == "ja"          # any kana -> ja
    assert detlang("ответ равен 42") == "ru"                   # Cyrillic -> ru
    assert detlang("উত্তর হল 42") == "bn"                      # Bengali -> bn


def test_detlang_strips_math_and_flags_undetectable():
    from scoring import detlang
    assert detlang("\\boxed{42}") == "?"                       # LaTeX/number only -> '?'
    assert detlang("42") == "?"
    assert detlang("2 + 2 = 4") == "?"                         # no prose


def test_detlang_latin_stays_in_eval_set():
    from scoring import detlang, _LATIN_LANGS
    out = detlang("The final answer to this word problem is forty two")
    assert out == "en"
    de = detlang("Die Antwort auf die Textaufgabe ist zweiundvierzig")
    assert de in _LATIN_LANGS or de == "?"                     # never a non-eval language
