"""Rate-limited progress lines with ETA for long GPU loops (SLURM/nbconvert-safe).

The E7 phases run under nbconvert with stdout buffered and redirected into
`results/scaleout_logs/<tag>.log`, so the only live signal is the per-phase sidecar log
(`results/E5_phase3_progress_<tag>.log` etc.) that the notebooks append to directly. This
module lets library code (evaluate.py batch loops, tuning.py searches) join that sidecar
without knowing which phase it is running in: the entry point exports `LRD_PROGRESS_FILE`
pointing at its sidecar, and every `Progress` here appends timestamped lines to it.

Emission is rate-limited (default one line per `LRD_PROGRESS_INTERVAL` seconds, 60): a loop
that finishes inside the first interval emits nothing, so the thousands of short dev evals a
λ-search makes stay silent while any long-running eval or search becomes visible with a
percentage and a linear-extrapolation ETA. Lines also go to stderr when it is a TTY (an
interactive salloc), never to a non-TTY stderr (the scaleout log already captures prints).
"""
from __future__ import annotations

import os
import sys
import time


def fmt_dur(s: float) -> str:
    """Compact duration: 47s / 3.2m / 1.75h."""
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


class Progress:
    """Counter with rate-limited ETA lines; see module docstring for routing.

    `total=None` counts without an ETA. `update(n, info=...)` advances and maybe emits;
    `close(info=...)` emits a final line only if the loop ever emitted (long enough to
    matter) or `force=True`. All I/O failures are swallowed — progress must never take
    down an eval.
    """

    def __init__(self, total: int | None, desc: str, *, min_interval: float | None = None,
                 stream=None):
        self.total = int(total) if total else None
        self.desc = desc
        fallback = 60.0 if min_interval is None else float(min_interval)
        try:
            env_iv = os.environ.get("LRD_PROGRESS_INTERVAL")
            self.min_interval = float(env_iv) if env_iv else fallback
        except ValueError:
            self.min_interval = fallback         # malformed env must not take down an eval
        self.n = 0
        self.t0 = time.monotonic()
        self._last = self.t0
        self._emitted = False
        self._path = os.environ.get("LRD_PROGRESS_FILE") or None
        self._stream = stream

    def update(self, n: int = 1, info: str = "") -> None:
        self.n += n
        now = time.monotonic()
        if now - self._last >= self.min_interval:
            self._emit(now, info)

    def close(self, info: str = "", force: bool = False) -> None:
        if self._emitted or force:
            self._emit(time.monotonic(), info or "done")

    def _emit(self, now: float, info: str = "") -> None:
        elapsed = now - self.t0
        if self.total:
            frac = self.n / max(1, self.total)
            eta = elapsed * (self.total - self.n) / max(1, self.n)
            msg = (f"{self.desc}: {self.n}/{self.total} ({100 * frac:.0f}%)  "
                   f"elapsed {fmt_dur(elapsed)}  ETA {fmt_dur(eta)}")
        else:
            msg = f"{self.desc}: {self.n}  elapsed {fmt_dur(elapsed)}"
        if info:
            msg += f"  {info}"
        self._last = now
        self._emitted = True
        line = f"{time.strftime('%H:%M:%S')}  {msg}\n"
        if self._path:
            try:
                with open(self._path, "a") as f:
                    f.write(line)
            except OSError:
                pass
        stream = self._stream if self._stream is not None else sys.stderr
        try:
            if hasattr(stream, "isatty") and stream.isatty():
                stream.write(line)
                stream.flush()
        except (OSError, ValueError):
            pass


def set_sidecar(path) -> None:
    """Export `path` as this process's progress sidecar (inherited by subprocesses)."""
    os.environ["LRD_PROGRESS_FILE"] = str(path)
