"""progress.Progress: rate-limited sidecar lines with ETA (LRD_PROGRESS_FILE routing)."""
from progress import Progress, fmt_dur


def test_emits_to_sidecar_with_eta(tmp_path, monkeypatch):
    side = tmp_path / "side.log"
    monkeypatch.setenv("LRD_PROGRESS_FILE", str(side))
    monkeypatch.delenv("LRD_PROGRESS_INTERVAL", raising=False)
    pr = Progress(10, "t", min_interval=0.0)          # 0 => every update emits
    pr.update(3, info="x")
    pr.update(7)
    pr.close()
    text = side.read_text()
    assert "t: 3/10 (30%)" in text and "ETA" in text and "x" in text
    assert "done" in text                             # close() emits after prior emission


def test_silent_when_loop_finishes_inside_interval(tmp_path, monkeypatch):
    side = tmp_path / "side.log"
    monkeypatch.setenv("LRD_PROGRESS_FILE", str(side))
    pr = Progress(5, "q", min_interval=3600.0)
    pr.update(5)
    pr.close()                                        # nothing emitted => close stays silent
    assert not side.exists()


def test_total_none_counts_without_eta(tmp_path, monkeypatch):
    side = tmp_path / "side.log"
    monkeypatch.setenv("LRD_PROGRESS_FILE", str(side))
    pr = Progress(None, "n", min_interval=0.0)
    pr.update(2)
    text = side.read_text()
    assert "n: 2" in text and "ETA" not in text


def test_fmt_dur():
    assert fmt_dur(30) == "30s"
    assert fmt_dur(90) == "1.5m"
    assert fmt_dur(7200) == "2.00h"
