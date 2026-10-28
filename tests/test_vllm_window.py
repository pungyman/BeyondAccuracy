"""The te empty-generation regression: vLLM 0.7 admits over-long TEXT prompts and finishes them
with 0 tokens instead of erroring, so an undersized max_model_len silently zeroes a language.
`check_context_window` is the fail-fast guard; these tests pin its contract (no vllm import,
no model — a stub tokenizer suffices because the guard only calls `.encode`).
"""
import pytest

from vllm_backend import check_context_window


class _StubTok:
    """encode() = 1 token per character; long prompts are just long strings."""
    def encode(self, text):
        return list(range(len(text)))


def test_fits_returns_longest():
    tok = _StubTok()
    longest = check_context_window(tok, ["x" * 100, "x" * 500],
                                   max_new_tokens=512, max_model_len=8192)
    assert longest == 500


def test_overflow_raises_with_required_size():
    tok = _StubTok()
    # 4944-token Telugu prompt vs the old 4096 window — the exact first-campaign failure
    with pytest.raises(ValueError, match=r"5456"):   # 4944 + 512 = required minimum
        check_context_window(tok, ["x" * 4944],
                             max_new_tokens=512, max_model_len=4096)


def test_boundary_exact_fit_passes():
    tok = _StubTok()
    check_context_window(tok, ["x" * (8192 - 512)],
                         max_new_tokens=512, max_model_len=8192)
