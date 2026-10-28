"""Unit tests for the measured-trough resolution in tuning.trough_layer (env-gated)."""
import json

import pytest

import tuning
from tuning import trough_layer


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("E5_TROUGH", raising=False)
    monkeypatch.delenv("E5_TROUGH_JSON", raising=False)


def _write_trough(tmp_path, model_tag, trough_layer_val, n=7500):
    p = tmp_path / f"trough_{model_tag}.json"
    p.write_text(json.dumps({"model_tag": model_tag, "trough_layer": trough_layer_val, "n": n}))
    return p


def test_no_env_dict_and_fallback_unchanged():
    assert trough_layer("Qwen2.5-1.5B-Instruct", 28) == 16
    assert trough_layer("Qwen2.5-7B-Instruct", 28) == 18
    assert trough_layer("Unlisted-Model", 30) == round(0.6 * 30)


def test_trough_json_overrides_dict(tmp_path, monkeypatch):
    p = _write_trough(tmp_path, "Qwen2.5-7B-Instruct", 20)
    monkeypatch.setenv("E5_TROUGH_JSON", str(p))
    assert trough_layer("Qwen2.5-7B-Instruct", 28) == 20


def test_trough_json_wrong_tag_ignored(tmp_path, monkeypatch):
    p = _write_trough(tmp_path, "Some-Other-Model", 20)
    monkeypatch.setenv("E5_TROUGH_JSON", str(p))
    assert trough_layer("Qwen2.5-7B-Instruct", 28) == 18  # falls through to the dict


def test_trough_json_out_of_range_ignored(tmp_path, monkeypatch):
    p = _write_trough(tmp_path, "Qwen2.5-7B-Instruct", 99)
    monkeypatch.setenv("E5_TROUGH_JSON", str(p))
    assert trough_layer("Qwen2.5-7B-Instruct", 28) == 18


def test_trough_json_missing_file_ignored(monkeypatch):
    monkeypatch.setenv("E5_TROUGH_JSON", "/nonexistent/trough.json")
    assert trough_layer("Qwen2.5-7B-Instruct", 28) == 18


def test_e5_trough_env_wins_over_json(tmp_path, monkeypatch):
    p = _write_trough(tmp_path, "Qwen2.5-7B-Instruct", 20)
    monkeypatch.setenv("E5_TROUGH_JSON", str(p))
    monkeypatch.setenv("E5_TROUGH", "7")
    assert trough_layer("Qwen2.5-7B-Instruct", 28) == 7


def test_e5_trough_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("E5_TROUGH", "28")
    with pytest.raises(ValueError):
        trough_layer("Qwen2.5-7B-Instruct", 28)


def test_source_logged_once_per_tag(capsys, monkeypatch):
    monkeypatch.setattr(tuning, "_TROUGH_LOGGED", set())
    trough_layer("Qwen2.5-1.5B-Instruct", 28)
    trough_layer("Qwen2.5-1.5B-Instruct", 28)
    out = capsys.readouterr().out
    assert out.count("[source: TROUGH_LAYER dict (n=512)]") == 1
