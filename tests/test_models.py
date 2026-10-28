"""Unit tests for the per-model registry + resolver (pure; no model, no network)."""
from models import (
    GenConfig,
    LADDER_13,
    MODEL_REGISTRY,
    ModelSpec,
    NON_THINKING_7,
    REASONING_6,
    angular_forced,
    model_spec,
)


def test_unknown_tag_resolves_to_defaults():
    # An unregistered tag must reduce to today's behavior exactly (greedy, thinking-off, 512-cap).
    s = model_spec("acme/Totally-Unknown-7B")
    assert s == ModelSpec()
    assert not s.is_reasoning and not s.gen.do_sample
    assert s.max_new_tokens is None and s.dtype is None and s.trust_remote_code is False


def test_non_reasoning_registered_stays_greedy():
    s = model_spec("Qwen/Qwen2.5-7B-Instruct")           # registered (angular metadata) but non-reasoning
    assert not s.is_reasoning and not s.gen.do_sample
    assert s.angular == "forced" and s.max_new_tokens is None


def test_qwen3_is_reasoning_with_sampling_and_cap():
    s = model_spec("Qwen/Qwen3-8B")
    assert s.is_reasoning and s.gen.do_sample
    assert s.gen.temperature == 0.6 and s.gen.top_p == 0.95 and s.gen.top_k == 20
    assert s.max_new_tokens == 16384


def test_deepseek_and_glm_sample_without_top_k():
    for tag in ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "THUDM/GLM-Z1-9B-0414"):
        s = model_spec(tag)
        assert s.is_reasoning and s.gen.do_sample
        assert s.gen.temperature == 0.6 and s.gen.top_p == 0.95 and s.gen.top_k is None


def test_r1_14b_loads_native_bf16():
    # E7/GB202: the 14B loads native bf16 (~30 GB) on the 96 GB card — the old fit="quant4bit" was a
    # 24 GB-A10G fit constraint, dropped so it matches the other 12 models' precision.
    assert model_spec("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B").fit == "single"


def test_yi_eos_and_internlm_trust_remote_code():
    assert model_spec("01-ai/Yi-1.5-9B-Chat").eos_token_id == 7
    assert not model_spec("01-ai/Yi-1.5-9B-Chat").is_reasoning
    assert model_spec("internlm/internlm2_5-7b-chat").trust_remote_code is True


def test_ladder_tiers():
    assert len(LADDER_13) == 13 and len(NON_THINKING_7) == 7 and len(REASONING_6) == 6
    assert set(NON_THINKING_7) | set(REASONING_6) == set(LADDER_13)
    # every reasoning ladder entry resolves to a reasoning spec; every non-thinking entry does not
    assert all(model_spec(m).is_reasoning for m in REASONING_6)
    assert all(not model_spec(m).is_reasoning for m in NON_THINKING_7)


def test_env_force_thinking_and_cap(monkeypatch):
    monkeypatch.setenv("E5_ENABLE_THINKING", "1")
    assert model_spec("Qwen/Qwen2.5-7B-Instruct").is_reasoning        # forced on globally
    monkeypatch.setenv("E5_ENABLE_THINKING", "0")
    assert not model_spec("Qwen/Qwen3-8B").is_reasoning               # forced off globally
    monkeypatch.delenv("E5_ENABLE_THINKING")
    monkeypatch.setenv("E5_REASON_MAX_NEW_TOKENS", "2048")
    assert model_spec("Qwen/Qwen3-8B").max_new_tokens == 2048         # reasoning cap override
    assert model_spec("Qwen/Qwen2.5-7B-Instruct").max_new_tokens is None  # non-reasoning untouched


def test_gen_config_defaults_are_greedy():
    g = GenConfig()
    assert g.do_sample is False and g.temperature is None and g.seed == 0


def test_angular_forced_gates_the_raw_diffpca_arm():
    # the 4 massive-activation models force the angular fit (raw DiffPCA collapses there)
    for tag in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
        assert angular_forced(tag)
    # every other tag (registered or not) runs raw as a fully-tuned sibling arm
    for tag in ("Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
                "google/gemma-2-2b-it", "acme/Totally-Unknown-7B"):
        assert not angular_forced(tag)
    # short tags resolve identically to full HF ids
    assert angular_forced("Qwen3-8B") and not angular_forced("Qwen2.5-1.5B-Instruct")
