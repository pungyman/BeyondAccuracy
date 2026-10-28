from __future__ import annotations
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from config import DATA_DIR, Config


def _resolve_parallel_file(lang: str, cfg: Config) -> Path:
    local_path = DATA_DIR / f"{lang}.json"
    if local_path.exists():
        return local_path

    if not cfg.data_repo:
        raise FileNotFoundError(
            f"Missing {local_path}. Run python scripts/fetch_data.py --revision <upstream-commit> "
            "or supply aligned language JSON files; see docs/reproduction.md."
        )

    from extract import _configure_hf_http

    _configure_hf_http()
    return Path(
        hf_hub_download(
            repo_id=cfg.data_repo,
            filename=f"{lang}.json",
            repo_type="dataset",
            revision=cfg.data_revision,
        )
    )


def _load_records(lang: str, cfg: Config) -> list[dict]:
    """Full parallel records (prompt + metadata) for one language."""
    with open(_resolve_parallel_file(lang, cfg)) as f:
        return json.load(f)


def stratified_indices(labels: Sequence, n: int, seed: int = 0) -> np.ndarray:
    """Proportional (largest-remainder) stratified sample of `n` indices into `labels`.

    Strata are the distinct values of `labels` (hashable; e.g. ``(level, type)`` tuples).
    Each stratum k receives ``round_lr(n * s_k / N)`` slots via the Hamilton/largest-remainder
    rule (floor everywhere, then hand the leftover one-by-one to the largest fractional
    remainders), so the per-stratum counts track the corpus shares and sum to exactly `n`.
    Within a stratum, members are drawn without replacement with a fixed RNG. Returns the
    selected indices as a **sorted** int array.
    """
    N = len(labels)
    if not 0 <= n <= N:
        raise ValueError(f"n={n} out of range for {N} records")
    rng = np.random.default_rng(seed)

    # group record indices by stratum, iterating strata in a stable sorted order so the
    # allocation/draw is reproducible regardless of input ordering
    groups: dict = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    strata = sorted(groups, key=str)
    sizes = np.array([len(groups[k]) for k in strata], dtype=np.int64)

    quota = n * sizes / N
    base = np.floor(quota).astype(np.int64)
    leftover = n - int(base.sum())
    if leftover:
        # largest fractional remainders win the leftover slots (ties -> larger stratum, then order)
        order = sorted(range(len(strata)), key=lambda k: (quota[k] - base[k], sizes[k]), reverse=True)
        for k in order[:leftover]:
            base[k] += 1

    picks: list[int] = []
    for k, count in zip(strata, base):
        idx = groups[k]
        if count >= len(idx):
            picks.extend(idx)                                # take all (count never exceeds size here)
        else:
            picks.extend(rng.choice(idx, size=int(count), replace=False).tolist())
    return np.sort(np.asarray(picks, dtype=np.int64))


def select_problem_indices(
    records: Sequence[dict],
    n: int,
    sample: str = "stratified",
    seed: int = 0,
    strata_keys: tuple[str, ...] = ("level", "type"),
) -> tuple[np.ndarray, list[dict]]:
    """Choose `n` problem indices and return them with their aligned metadata labels.

    - ``sample="stratified"``: proportional (level x type) sample (`stratified_indices`).
    - ``sample="first"``: legacy ``arange(n)`` (the original first-n slice).

    `labels` is the per-selected-row metadata (`strata_keys` + ``outcome`` when present),
    ordered to match the returned indices.
    """
    if sample == "stratified":
        labels = [tuple(r.get(k) for k in strata_keys) for r in records]
        indices = stratified_indices(labels, n, seed)
    elif sample == "first":
        if n > len(records):
            raise ValueError(f"n={n} exceeds {len(records)} records")
        indices = np.arange(n, dtype=np.int64)
    else:
        raise ValueError(f"unknown sample mode {sample!r} (expected 'stratified' or 'first')")

    keys = (*strata_keys, "outcome")
    row_labels = [{k: records[i].get(k) for k in keys} for i in indices]
    return indices, row_labels


def load_parallel(
    langs,
    n: int,
    prompt_key: str = "translated_prompt",
    cfg: Config | None = None,
    *,
    indices: Sequence[int] | None = None,
) -> dict[str, list[str]]:
    """Index-aligned prompts per language.

    By default the `n` problems are chosen by `cfg.sample` (stratified proportional sampling
    over `cfg.strata_keys`, seeded by `cfg.seed`); pass `indices` to reuse a precomputed
    selection (e.g. the one persisted next to the activation cache). The same indices are used
    for every language so the corpus stays parallel.
    """
    cfg = cfg or Config()
    langs = list(langs)
    records = {lang: _load_records(lang, cfg) for lang in langs}
    lengths = {len(r) for r in records.values()}
    assert len(lengths) == 1, f"non-parallel corpus lengths: {lengths}"

    if indices is None:
        indices, _ = select_problem_indices(
            records[langs[0]], n, sample=cfg.sample, seed=cfg.seed, strata_keys=cfg.strata_keys
        )
    return {lang: [records[lang][i][prompt_key] for i in indices] for lang in langs}


def build_chat_inputs(tokenizer, prompts: list[str], *, enable_thinking: bool = False) -> list[str]:
    """Apply the default chat template (add_generation_prompt=True), no system override.

    `enable_thinking` gates Qwen3-style `<think>` traces. Default **False** (extraction and
    non-reasoning eval): reasoning models then emit a direct answer, comparable to non-thinking
    families and not truncated mid-thought by the generation cap. Pass **True** for thinking-ON
    MGSM eval of reasoning models (Zhao et al. protocol). The flag is ignored by templates that
    don't use it; templates that *reject* the kwarg raise `TypeError` and fall back (this is also
    the GLM-Z1/R1 case, whose templates always emit `<think>` regardless — the `</think>` strip in
    `scoring.strip_reasoning` is the real safety net there).
    """
    def _tmpl(p: str) -> str:
        msg = [{"role": "user", "content": p}]
        try:
            return tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return [_tmpl(p) for p in prompts]
