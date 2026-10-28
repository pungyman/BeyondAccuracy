"""Answer extraction / matching + output-language detection for behavioral experiments.

Promoted from scripts/e10_behavioral.py so E10 and the cross-language validation (E12) share
one tested implementation. Matching is intentionally lenient but applied identically across
conditions, so unbiased scorer noise cancels in paired comparisons.
"""
from __future__ import annotations
import re
import unicodedata
from fractions import Fraction

from prompts import MGSM_ANSWER_PREFIX, MMLU_ANSWER_CUE

try:
    from langdetect import detect, detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:  # pragma: no cover - langdetect optional
    _HAS_LANGDETECT = False

# Final-answer instruction (\boxed{} is language-neutral notation; the prose is localized so a
# Japanese prompt stays fully Japanese and does not prime English output).
BOXED_INSTRUCTION = {
    "en": "\n\nGive the final answer in \\boxed{}.",
    "ja": "\n\n最終的な答えを \\boxed{} の形式で示してください。",
    "zh": "\n\n请将最终答案写成 \\boxed{} 的形式。",
}


def make_math_prompt(problem: str, instr_lang: str = "ja") -> str:
    """Append a final-answer instruction in `instr_lang` to a problem statement."""
    return problem + BOXED_INSTRUCTION[instr_lang]


def extract_pred(text: str) -> str:
    """Pull the model's final answer: prefer \\boxed{...}, then 'answer is/=' cues, then last number."""
    boxes = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if boxes:
        return boxes[-1]
    m = re.findall(r'(?:answer is|answer:|equals|结果是|答案是|答えは|=)\s*\$?\s*([\-0-9][\-0-9.,/]*)', text, re.I)
    if m:
        return m[-1]
    nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    return nums[-1] if nums else ""


def norm(s) -> str:
    """Normalize an answer string for comparison (strip LaTeX wrappers, spaces, commas)."""
    s = str(s)
    s = re.sub(r'\\boxed|\\text|\\left|\\right|\\!|\\,|\\ |\$|%|\\dfrac|\\frac', '', s)
    s = s.replace('{', '').replace('}', '').replace(' ', '').replace(',', '').strip().rstrip('.').lower()
    return s


def is_correct(pred, gold) -> bool:
    """True if pred matches gold exactly (normalized) or numerically (handles fractions)."""
    p, g = norm(pred), norm(gold)
    if g == "":
        return False
    if p == g:
        return True
    try:
        return abs(float(Fraction(p)) - float(Fraction(g))) < 1e-6
    except Exception:
        return False


# --- Benchmark scorers (E5): multiple-choice, cloze, and the native MGSM numeric path ---------

def _ascii_digits(s: str) -> str:
    """Map every Unicode decimal digit to ASCII 0–9, leaving other characters untouched.

    Covers Bengali (`০–৯`), Telugu (`౦–౯`), Thai (`๐–๙`), Devanagari, Arabic-Indic, etc. via
    `unicodedata.decimal`, so the numeric MGSM path works on non-Latin scripts (`bn te th`).
    """
    out = []
    for c in s:
        d = unicodedata.decimal(c, None)
        out.append(str(d) if d is not None else c)
    return "".join(out)


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text: str) -> str:
    """Last number in `text` (native digits already normalized), commas stripped; '' if none."""
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else ""


_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def strip_reasoning(text: str, *, expect_trace: bool = False) -> str:
    """Drop a reasoning trace before answer extraction: everything up to & incl. the LAST `</think>`.

    Reasoning models (Qwen3/GLM-Z1/R1-distill) emit `<think>...</think>` then the answer; scoring
    the raw text would grab a number from the trace. Rules:
      - has `</think>`  -> return the tail after the last one (handles nested/repeated tags);
      - no `</think>` but a trace was expected/opened -> "" (truncated trace, no answer emitted; do
        NOT score a mid-reasoning number). `expect_trace=True` covers the **forced-`<think>` models**
        (GLM-Z1, R1-distill) whose opening `<think>` is emitted PROMPT-side, so the decoded
        continuation carries neither tag on a truncated trace; `<think>` present in the text covers
        Qwen3 (output-side tag);
      - no think tags and no trace expected (every non-reasoning output) -> unchanged (a no-op).
    """
    end = text.rfind(_THINK_CLOSE)
    if end != -1:
        return text[end + len(_THINK_CLOSE):]
    return "" if (expect_trace or _THINK_OPEN in text) else text


def extract_mgsm_answer(text: str, lang: str) -> str:
    """Native-route MGSM extraction: split on the localized answer keyword, take the last number.

    The prompt cues the model to end with "<localized keyword>: <number>" (see
    `prompts.MGSM_ANSWER_PREFIX`), so we prefer the number after the *last* occurrence of that
    keyword; failing that we fall back to the last number anywhere. Native digits (Bengali /
    Telugu / Thai) are normalized to ASCII first so `is_correct` compares numerically.
    """
    norm_text = _ascii_digits(text)
    prefix = MGSM_ANSWER_PREFIX.get(lang)
    if prefix:
        idx = norm_text.rfind(prefix)
        if idx != -1:
            tail = norm_text[idx + len(prefix):]
            num = _last_number(tail)
            if num:
                return num
    return _last_number(norm_text)


_CHOICE_LETTERS = "ABCDEFGH"


def extract_choice(text: str, n_choices: int = 4) -> str:
    """The model's chosen MC letter: prefer one right after a localized "Answer:" cue, else the
    last standalone letter in the valid range. Returns "" if none is found.
    """
    letters = _CHOICE_LETTERS[:n_choices]
    cues = {c.rstrip(":") for c in MMLU_ANSWER_CUE.values()} | {"Answer"}
    # a letter immediately following any localized answer cue wins (e.g. "答案: B", "Answer: C")
    for cue in cues:
        m = re.findall(rf"{re.escape(cue)}\s*[:：]?\s*\(?([{letters}])\)?\b", text)
        if m:
            return m[-1].upper()
    standalone = re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", text)
    return standalone[-1].upper() if standalone else ""


def is_correct_mc(pred_letter: str, gold_letter: str) -> bool:
    """Case-insensitive exact letter match; empty prediction is always wrong."""
    p = (pred_letter or "").strip().upper()
    g = (gold_letter or "").strip().upper()
    return bool(p) and p == g


def score_cloze(option_logprobs: tuple[float, float]) -> str:
    """Argmax of the two XWinograd continuation log-probs -> "1" / "2"."""
    return "1" if option_logprobs[0] >= option_logprobs[1] else "2"


def script_dom(s: str) -> str:
    """Dominant writing system among alphabetic chars: Han / Kana / Latin / Cyrillic / ... / none."""
    b = {"Han": 0, "Kana": 0, "Latin": 0, "Cyrillic": 0, "Bengali": 0, "Telugu": 0, "Thai": 0, "other": 0}
    tot = 0
    for c in s:
        if c.isspace() or c.isdigit() or not c.isalpha():
            continue
        o = ord(c)
        tot += 1
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
            b["Han"] += 1
        elif 0x3040 <= o <= 0x30FF:
            b["Kana"] += 1
        elif (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A) or 0x00C0 <= o <= 0x024F:
            b["Latin"] += 1
        elif 0x0400 <= o <= 0x04FF:
            b["Cyrillic"] += 1
        elif 0x0980 <= o <= 0x09FF:
            b["Bengali"] += 1
        elif 0x0C00 <= o <= 0x0C7F:
            b["Telugu"] += 1
        elif 0x0E00 <= o <= 0x0E7F:
            b["Thai"] += 1
        else:
            b["other"] += 1
    return max(b, key=b.get) if tot else "none"


# Output-language detection, hardened for the MGSM fidelity axis (E7 gap #5): naive langdetect
# misclassifies math-heavy / short / LaTeX-laden CoTs (verified ~17pp deflation on real Chinese
# outputs). We (a) strip LaTeX & math, (b) resolve script-distinguishable languages by SCRIPT
# (unambiguous for 6 of 11), and (c) only fall back to langdetect — restricted to the 5 Latin eval
# languages — for genuinely Latin prose, returning '?' when there is too little signal to decide.
_EVAL_LANGS = frozenset({"bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"})
_LATIN_LANGS = frozenset({"de", "en", "es", "fr", "sw"})
_SCRIPT_LANG = {"Cyrillic": "ru", "Bengali": "bn", "Telugu": "te", "Thai": "th"}  # unambiguous
# strip LaTeX commands, delimiters, and standalone numbers so detection sees prose, not markup
_MATH_STRIP = re.compile(r"\\[a-zA-Z]+|\\.|[${}\[\]()^_~|]|\b\d[\d.,/]*\b")


def _has_kana(s: str) -> bool:
    return any(0x3040 <= ord(c) <= 0x30FF for c in s)


def detlang(s: str) -> str:
    """Detected output language among the 11 eval languages, or '?' if undetectable.

    Script-aware and LaTeX-stripped (see the note above). '?' means "no usable language signal"
    (e.g. digits-only, or too-short Latin prose) — callers should EXCLUDE '?' from the fidelity
    denominator rather than score it as a mismatch (E7 gap #5).
    """
    prose = _MATH_STRIP.sub(" ", s)
    if _has_kana(prose):                                   # any kana ⇒ Japanese (kanji-only ⇒ zh below)
        return "ja"
    sd = script_dom(prose)
    if sd in _SCRIPT_LANG:
        return _SCRIPT_LANG[sd]
    if sd == "Han":
        return "zh"
    if sd != "Latin" or not _HAS_LANGDETECT:
        return "?"
    if len(re.sub(r"[^a-zA-Z]", "", prose)) < 8:           # too little Latin prose to decide
        return "?"
    try:
        for c in detect_langs(prose):                      # ranked; take the top candidate in-set
            code = {"zh-cn": "zh", "zh-tw": "zh"}.get(c.lang, c.lang)
            if code in _LATIN_LANGS:
                return code
    except Exception:
        return "?"
    return "?"


def is_english(text: str) -> bool:
    """Coarse 'did the output come out in English/Latin script?' signal (script-based, robust to math)."""
    return script_dom(text) == "Latin"


# Common English function/math words — a cheap fluency proxy that separates real English from
# Latin-script gibberish (e.g. a random-direction steer that emits "Yuzo-123-000..." is Latin
# script but has ~zero of these; genuine reasoning "to solve this problem we need..." has many).
_COMMON_EN = frozenset(
    "the to of and a is we this that in for be are need number find so then each sum total "
    "answer solve problem ways can with by have first consider two how many it as on".split()
)


def english_word_hits(text: str) -> int:
    """Count of distinct common English words present (0 for gibberish, high for fluent prose)."""
    return len(set(re.findall(r"[a-z]+", text.lower())) & _COMMON_EN)


def coherent_english(text: str, min_hits: int = 4) -> bool:
    """True only if the output is Latin script AND reads as actual English (>= min_hits common words).

    Stricter than is_english(): rejects Latin-script gibberish, so it isolates *coherent* English
    output from a mere script flip. Use as the primary 'did injection produce English?' signal.
    """
    return is_english(text) and english_word_hits(text) >= min_hits
