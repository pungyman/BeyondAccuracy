#!/usr/bin/env python
"""Smoke for the te empty-generation fix: Qwen2.5-7B on vLLM with the new default window (8192)
must produce non-empty Telugu generations where the 4096 window produced 0-token rows.
Reads nothing from and writes nothing to results/ — prints a verdict line for the log.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from benchmarks import _load_mgsm
from vllm_backend import VllmRunner, run_eval_vllm, check_context_window
from corpus import build_chat_inputs
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
items = _load_mgsm("te", "test", n_shot=8)[:6]
tok = AutoTokenizer.from_pretrained(MODEL)
prompts = build_chat_inputs(tok, [it.prompt for it in items])
runner = VllmRunner(MODEL)   # new default max_model_len=8192
longest = check_context_window(tok, prompts, max_new_tokens=512, max_model_len=runner.max_model_len)
print(f"[smoke] longest te prompt {longest} tok, window {runner.max_model_len}", flush=True)
df = run_eval_vllm(runner, tok, items, cond="clean", max_new_tokens=512, model_name=MODEL)
print(df[["id", "n_new_tokens", "truncated", "pred", "gold", "correct"]].to_string(index=False), flush=True)
empty = int((df.n_new_tokens == 0).sum())
print(f"[smoke] VERDICT: {'PASS' if empty == 0 else 'FAIL'} — {empty}/{len(df)} empty generations, "
      f"{int(df.correct.sum())}/{len(df)} correct", flush=True)
sys.exit(0 if empty == 0 else 1)
