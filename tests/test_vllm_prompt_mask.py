"""The `prompt_only` seam under continuous batching — `open.md` Q3.

vLLM 0.8.5's V1 engine forces chunked prefill on, so "is there a zero in `positions`?" stops being
a sound test for "is this forward a prompt pass": a continuation chunk carries no zero and would be
skipped as if it were decode. These tests pin the replacement, `prompt_token_mask`, which decides
per TOKEN from the engine's own batch metadata. No vllm import and no model — stub metadata objects
carry the two layouts (V0's prefill-first split point, V1's `query_start_loc` + prompt lengths).

The case worth naming is `test_v1_final_single_token_chunk_is_prompt`: a chunked prefill's last
chunk can be one token, which is the same SHAPE as a decode step. Only the position tells them
apart, which is why the mask compares positions against prompt lengths rather than query lengths.
"""
import pytest
import torch

from vllm_backend import _apply_edit, _layer_hook, prompt_token_mask


class _V0Meta:
    """vLLM 0.7: the batch is prefill tokens first, then one token per decoding sequence."""
    def __init__(self, num_prefill_tokens):
        self.num_prefill_tokens = num_prefill_tokens


class _V1Meta:
    """vLLM >=0.8: requests flattened in `input_batch` order, boundaries in `query_start_loc`."""
    def __init__(self, query_lens):
        self.query_start_loc = torch.tensor([0] + list(torch.cumsum(
            torch.tensor(query_lens), 0)), dtype=torch.int32)


def test_v0_split_point_is_exact():
    mask = prompt_token_mask(torch.arange(8), _V0Meta(num_prefill_tokens=5))
    assert mask.tolist() == [True] * 5 + [False] * 3


def test_v0_padding_beyond_the_batch_is_not_edited():
    # 3 prefill + 1 decode token, then CUDA-graph padding out to 8.
    mask = prompt_token_mask(torch.zeros(8, dtype=torch.long), _V0Meta(num_prefill_tokens=3))
    assert mask.tolist() == [True] * 3 + [False] * 5


def test_v1_unsplit_prefill_is_all_prompt():
    mask = prompt_token_mask(torch.arange(4), _V1Meta([4]), num_prompt_tokens=[4])
    assert mask.tolist() == [True] * 4


def test_v1_middle_chunk_has_no_zero_but_is_still_prompt():
    """The regression: chunked prefill, positions 4..6 of a 10-token prompt, no zero in sight."""
    mask = prompt_token_mask(torch.tensor([4, 5, 6]), _V1Meta([3]), num_prompt_tokens=[10])
    assert mask.tolist() == [True] * 3


def test_v1_final_single_token_chunk_is_prompt():
    """Query length 1 — shaped exactly like a decode step, and it must still be edited."""
    mask = prompt_token_mask(torch.tensor([9]), _V1Meta([1]), num_prompt_tokens=[10])
    assert mask.tolist() == [True]


def test_v1_decode_step_is_not_prompt():
    """Same shape as the case above; only the position (10, not 9) separates them."""
    mask = prompt_token_mask(torch.tensor([10]), _V1Meta([1]), num_prompt_tokens=[10])
    assert mask.tolist() == [False]


def test_v1_mixed_batch_splits_per_request():
    """One request mid-prefill next to two decoding — the batch continuous batching produces."""
    positions = torch.tensor([4, 5, 6, 12, 30])          # req0 chunk, req1 decode, req2 decode
    mask = prompt_token_mask(positions, _V1Meta([3, 1, 1]), num_prompt_tokens=[10, 8, 30])
    assert mask.tolist() == [True, True, True, False, False]


def test_v1_padding_tail_is_never_edited():
    positions = torch.tensor([0, 1, 2, 0, 0])            # 3 real tokens, then graph padding
    mask = prompt_token_mask(positions, _V1Meta([3]), num_prompt_tokens=[10])
    assert mask.tolist() == [True, True, True, False, False]


@pytest.mark.parametrize("md,npt", [
    (None, [4]),                                          # no metadata at all
    (_V1Meta([4]), None),                                 # V1 without input_batch reachable
])
def test_returns_none_rather_than_guessing(md, npt):
    assert prompt_token_mask(torch.arange(4), md, npt) is None


def test_none_when_metadata_is_wider_than_the_prompt_lengths():
    """Three requests in the batch, two prompt lengths — a mask here would be misaligned."""
    assert prompt_token_mask(torch.arange(3), _V1Meta([1, 1, 1]), num_prompt_tokens=[4, 4]) is None


def test_apply_edit_falls_back_to_all_or_nothing_on_a_shape_mismatch():
    stream = torch.ones(4, 2)
    zero = lambda h: torch.zeros_like(h)                  # noqa: E731 - one-line stub
    assert torch.equal(_apply_edit(stream, zero, torch.tensor([True])), torch.zeros(4, 2))
    assert torch.equal(_apply_edit(stream, zero, torch.tensor([False])), stream)


def test_hook_edits_only_the_prompt_rows_of_a_mixed_batch():
    """End to end through the deferred-residual layout: `hidden + residual` is the stream."""
    hidden, residual = torch.ones(3, 2), torch.ones(3, 2)      # stream = 2.0 everywhere
    mask = torch.tensor([True, False, True])
    hook = _layer_hook(lambda h: torch.zeros_like(h), "prompt_only", mask_fn=lambda p: mask)
    out = hook(None, (torch.tensor([0, 9, 1]),), (hidden, residual))
    edited_stream = out[0] + residual
    assert edited_stream[0].tolist() == [0.0, 0.0]             # ablated
    assert edited_stream[1].tolist() == [2.0, 2.0]             # decode row, untouched
    assert edited_stream[2].tolist() == [0.0, 0.0]


def test_hook_skips_a_batch_with_no_prompt_token():
    hidden, residual = torch.ones(2, 2), torch.ones(2, 2)
    hook = _layer_hook(lambda h: torch.zeros_like(h), "prompt_only",
                       mask_fn=lambda p: torch.tensor([False, False]))
    out = hook(None, (torch.tensor([9, 10]),), (hidden, residual))
    assert out[0] is hidden                                    # returned unchanged, not recomputed


def test_hook_without_metadata_falls_back_to_the_zero_heuristic():
    hidden, residual = torch.ones(2, 2), torch.ones(2, 2)
    hook = _layer_hook(lambda h: torch.zeros_like(h), "prompt_only", mask_fn=lambda p: None)
    assert hook(None, (torch.tensor([7, 8]),), (hidden, residual))[0] is hidden    # no zero: decode
    edited = hook(None, (torch.tensor([0, 1]),), (hidden, residual))[0]
    assert torch.equal(edited + residual, torch.zeros(2, 2))                      # zero: prompt
