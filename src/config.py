from __future__ import annotations
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]   # repo root (.../LRD)
DATA_DIR = REPO / "data" / "parallel_multilingual_math"
ACTS_DIR = REPO / "data" / "activations"
RESULTS_DIR = REPO / "results"
FIGS_DIR = REPO / "figs"

LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]  # alphabetical, matches notebook 01

# --- results layout -------------------------------------------------------
# Artifacts live at results/<experiment>/<model_tag>/<filename>; filenames keep
# their historical flat form, so only the placement is new. Route every write
# through results_path() and every read-back through results_glob() — a bare
# RESULTS_DIR / name writes to the root, where only DIGEST.md belongs.
SHARED_SUBDIR = "shared"       # cross-experiment per-model inputs (trough_<tag>.json)

# Bootstrap set for inferring a tag from a filename; results_path() itself never
# guesses (callers pass model_tag), so this only backstops tidy_results.py, which
# also picks up any <model_tag> dir already present under results/.
MODEL_TAGS = (
    "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-8B",
    "gemma-2-2b-it", "gemma-3-12b-it", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3",
)

# Anchored so the E7 in "E5_E7_fixedhth_..." is seen: those files are written by
# the E5 benchmark harness but are E7 work, and the highest number wins.
_EXP_RE = re.compile(r"(?<![A-Za-z0-9])E(\d+)_")


def experiment_of(name: str) -> str | None:
    """`E<n>` bucket for a results filename — the highest experiment it names."""
    hits = _EXP_RE.findall(name)
    return f"E{max(int(h) for h in hits)}" if hits else None


def model_tag_of(name: str, known: tuple[str, ...] | None = None) -> str | None:
    """Longest known model tag appearing in `name` (longest wins so that e.g.
    Qwen2.5-7B-Instruct is not shadowed by a shorter prefix)."""
    hits = [t for t in (known or MODEL_TAGS) if t in name]
    return max(hits, key=len) if hits else None


def results_path(name: str, model_tag: str | None = None, *, mkdir: bool = True) -> Path:
    """Route a results filename to results/<experiment>/<model_tag>/<name>.

    `model_tag` is explicit because every caller already knows it; when omitted
    it is inferred from `name`. Files with no experiment prefix (trough_<tag>.json)
    land in results/<SHARED_SUBDIR>/; anything unroutable stays at the root.
    """
    tag = model_tag or model_tag_of(name)
    exp = experiment_of(name)
    if exp and tag:
        out = RESULTS_DIR / exp / tag / name
    elif tag:
        out = RESULTS_DIR / SHARED_SUBDIR / name
    else:
        out = RESULTS_DIR / name
    if mkdir:
        out.parent.mkdir(parents=True, exist_ok=True)
    return out


def results_glob(pattern: str) -> list[Path]:
    """Sorted matches for a flat-form filename `pattern` under the routed layout.

    Searches every routed bucket, so a caller that knows only the filename shape
    does not have to reconstruct the bucket. Deliberately does NOT descend into
    the frozen run bundles (archive_pilot_*, partial_*): they hold superseded
    copies under identical basenames, and a plain rglob would silently mix them
    into a live read.
    """
    roots = [d for d in RESULTS_DIR.iterdir()
             if d.is_dir() and (_EXP_RE.match(f"{d.name}_") or d.name == SHARED_SUBDIR)]
    return sorted(p for root in roots for p in root.rglob(pattern))

@dataclass(frozen=True)
class Config:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_repo: str = os.environ.get("LRD_DATA_REPO", "")
    data_revision: str = "main"
    acts_repo: str = os.environ.get("LRD_ACTS_REPO", "")
    acts_revision: str = "main"
    langs: tuple[str, ...] = tuple(LANGS)
    n_samples: int = 32
    prompt_key: str = "translated_prompt"
    seed: int = 0
    sample: str = "stratified"                # "stratified" (proportional level x type) or "first" (legacy data[:n])
    strata_keys: tuple[str, ...] = ("level", "type")  # record fields defining strata for stratified sampling
    intervene_layer: int | None = None      # None -> resolved to n_layers//2 at runtime
    dtype: torch.dtype = torch.bfloat16

    @property
    def n_langs(self) -> int:
        return len(self.langs)

    @property
    def rank(self) -> int:
        return self.n_langs - 1

def set_seeds(seed: int = 0) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

LINALG_DEVICE = torch.device("cpu")   # SVD on MPS is unreliable
