"""Attributable localized prompt literals + pure prompt builders for the E5 benchmarks.

The MGSM path is the **native / attributable** route (E5 P1): rather than appending a
hand-invented ``\\boxed{}`` instruction in every language, MGSM reuses the dataset's own
localized few-shot exemplars and a *vendored, cited* localized answer keyword. The scorer
(`scoring.extract_mgsm_answer`) splits the model output on that same keyword, so the two must
stay in lockstep.

All builders are **pure** (str/dict in, str out — no model, no tokenizer), so they are
unit-testable offline with tiny inline fixtures; chat-templating happens later in the harness.
"""
from __future__ import annotations

# --- MGSM localized answer keyword ---------------------------------------------------------
# Source: the MGSM benchmark's canonical ``LANG_TO_ANSWER_PREFIX`` table (Shi et al. 2022,
# "Language Models are Multilingual Chain-of-Thought Reasoners", arXiv:2210.03057; released in
# google-research/url-nlp and mirrored in OpenAI simple-evals / lighteval `translation_literal`).
# The keyword terminates each exemplar's worked solution ("<keyword>: <number>") and is what we
# split the model's own generation on. Do NOT machine-translate these — they are attributable.
MGSM_ANSWER_PREFIX: dict[str, str] = {
    "en": "Answer",
    "es": "Respuesta",
    "fr": "Réponse",
    "de": "Antwort",
    "ru": "Ответ",
    "zh": "答案",
    "ja": "答え",
    "th": "คำตอบ",
    "sw": "Jibu",
    "bn": "উত্তর",
    "te": "సమాధానం",
}

# --- M-MMLU localized "Answer" cue ---------------------------------------------------------
# The multiple-choice context ends with a localized "Answer:" cue before the scored letter,
# mirroring lighteval's multilingual MMLU templates (which localize the "Answer" literal via the
# same `translation_literal.answer` word). Global-MMLU covers 10 of our 11 (no `th`), so no Thai
# entry is required, but we reuse the MGSM keyword for `th` too for robustness / completeness.
MMLU_ANSWER_CUE: dict[str, str] = {lang: f"{word}:" for lang, word in MGSM_ANSWER_PREFIX.items()}


def _instr_lang(item_lang: str, instr_lang: str | None) -> str:
    """Instruction language defaults to the item's own language (never prime English)."""
    return instr_lang if instr_lang is not None else item_lang


def build_mgsm_prompt(
    item: dict,
    exemplars: list[dict] | None = None,
    *,
    n_shot: int = 0,
    instr_lang: str | None = None,
) -> str:
    """Native MGSM few-shot prompt: `n_shot` localized exemplars then the test question.

    `item` is the test record (`{"question": ...}`); `exemplars` are localized `train` records
    (`{"question": ..., "answer": <localized CoT>}`). Each exemplar is rendered as its question
    followed by its worked CoT answer (which itself ends with the localized answer keyword). The
    test question is appended with a trailing answer cue so the model continues in the same
    "<keyword>: <number>" format the scorer parses. No `\\boxed{}` instruction is added.

    Pure: does not touch a model/tokenizer. Chat-templating is applied later by the harness.
    """
    lang = item.get("lang") or _instr_lang("en", instr_lang)
    prefix = MGSM_ANSWER_PREFIX.get(lang, MGSM_ANSWER_PREFIX["en"])
    exemplars = list(exemplars or [])[:n_shot]

    blocks: list[str] = []
    for ex in exemplars:
        q = str(ex["question"]).strip()
        a = str(ex.get("answer", "")).strip()
        blocks.append(f"{q}\n{a}")
    blocks.append(f"{str(item['question']).strip()}\n{prefix}:")
    return "\n\n".join(blocks)


def build_mmlu_prompt(item: dict, *, instr_lang: str | None = None) -> str:
    """M-MMLU log-likelihood context: question + localized A–D block + localized answer cue.

    The scored continuations are the letters "A".."D" (built by the loader into
    `BenchItem.choices`); this returns only the shared context they are appended to.
    """
    lang = item.get("lang") or _instr_lang("en", instr_lang)
    cue = MMLU_ANSWER_CUE.get(lang, MMLU_ANSWER_CUE["en"])
    q = str(item["question"]).strip()
    letters = ["A", "B", "C", "D"]
    options = [str(item[f"option_{l.lower()}"]).strip() for l in letters]
    block = "\n".join(f"{l}. {opt}" for l, opt in zip(letters, options))
    return f"{q}\n{block}\n{cue}"


def build_xwinograd_continuations(item: dict) -> tuple[str, str]:
    """Substitute each option into the `_` blank of an XWinograd sentence.

    Returns `(full_with_option1, full_with_option2)` — the two full continuations scored by
    log-likelihood. No instruction text: XWinograd is pure continuation scoring.
    """
    sentence = str(item["sentence"])
    opt1, opt2 = str(item["option1"]), str(item["option2"])
    return sentence.replace("_", opt1, 1), sentence.replace("_", opt2, 1)
