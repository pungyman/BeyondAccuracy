"""Unit tests for the spec-driven model-load kwargs (pure; no network / no from_pretrained)."""
import torch

from config import Config
from extract import _model_load_kwargs
from models import ModelSpec


def test_pinned_load_preserves_dtype_on_old_and_new_transformers(monkeypatch):
    from unittest.mock import MagicMock, patch
    import extract
    import transformers

    monkeypatch.setenv('LRD_MODEL_REVISION', 'a' * 40)
    monkeypatch.setattr(extract, '_configure_hf_http', lambda: None)
    for version, dtype_key in [('4.51.3', 'torch_dtype'), ('5.14.1', 'dtype')]:
        monkeypatch.setattr(transformers, '__version__', version)
        tokenizer = MagicMock()
        tokenizer.pad_token = 'pad'
        with patch.object(extract.AutoTokenizer, 'from_pretrained', return_value=tokenizer) as tok_load, \
             patch.object(extract.AutoModelForCausalLM, 'from_pretrained') as model_load:
            extract.load_model_and_tokenizer(Config(), 'cpu', spec=ModelSpec())
        assert tok_load.call_args.kwargs['revision'] == 'a' * 40
        assert model_load.call_args.kwargs['revision'] == 'a' * 40
        assert model_load.call_args.kwargs[dtype_key] == Config().dtype
        assert ('dtype' if dtype_key == 'torch_dtype' else 'torch_dtype') not in model_load.call_args.kwargs


def test_default_spec_is_backward_compatible():
    cfg = Config()
    tok_kwargs, model_kwargs = _model_load_kwargs(cfg, ModelSpec())
    assert tok_kwargs == {"trust_remote_code": False}
    assert model_kwargs["dtype"] == cfg.dtype                 # falls back to cfg.dtype (bf16)
    assert model_kwargs["trust_remote_code"] is False
    assert "attn_implementation" not in model_kwargs          # not injected unless set


def test_trust_remote_code_threads_to_both():
    cfg = Config()
    tok_kwargs, model_kwargs = _model_load_kwargs(cfg, ModelSpec(trust_remote_code=True))
    assert tok_kwargs["trust_remote_code"] is True
    assert model_kwargs["trust_remote_code"] is True


def test_dtype_override_wins_over_cfg():
    cfg = Config()
    _, model_kwargs = _model_load_kwargs(cfg, ModelSpec(dtype=torch.float16))
    assert model_kwargs["dtype"] == torch.float16


def test_attn_implementation_only_when_set():
    cfg = Config()
    _, mk_none = _model_load_kwargs(cfg, ModelSpec())
    _, mk_set = _model_load_kwargs(cfg, ModelSpec(attn_implementation="flash_attention_2"))
    assert "attn_implementation" not in mk_none
    assert mk_set["attn_implementation"] == "flash_attention_2"


def test_quant4bit_fit_attaches_4bit_config_else_none():
    cfg = Config()
    _, mk_quant = _model_load_kwargs(cfg, ModelSpec(fit="quant4bit"))
    qc = mk_quant.get("quantization_config")
    assert qc is not None and getattr(qc, "load_in_4bit", False)
    _, mk_plain = _model_load_kwargs(cfg, ModelSpec())              # default fit="single"
    assert "quantization_config" not in mk_plain                    # backward compat
