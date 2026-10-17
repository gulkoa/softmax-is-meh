#!/usr/bin/env python3
"""Tests for scripts/early_cancel_nan.py (CPU, no data, no squeue).

    python scripts/test_early_cancel_nan.py
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from early_cancel_nan import nan_runs  # noqa: E402
from test_stability_flags import log  # noqa: E402


def setup(chunks):
    """chunks: {name: [log path or None (not started), ...]} -> jobs file."""
    d = tempfile.mkdtemp()
    lines, j = [], 100
    for name, paths in chunks.items():
        ids = []
        for p in paths:
            j += 1
            ids.append(str(j))
            if p is not None:
                shutil.copy(p, os.path.join(d, f"qknorm-pilot-{j}.out"))
        lines.append(f"{name} {' '.join(ids)}")
    jf = os.path.join(d, "jobs.txt")
    open(jf, "w").write("\n".join(lines) + "\n\n")
    return jf, d


def test_detects_nan_and_lists_all_jobs():
    nan = log(last=6000, gn=lambda s: float("nan") if s >= 5000 else 0.5)
    jf, d = setup({"A@2e-3-s1": [nan, None, None], "A@4e-3-s1": [log(last=3000), None]})
    r = {n: (b, jobs) for n, b, jobs in nan_runs(jf, d)}
    assert r["A@2e-3-s1"] == (5000, ["101", "102", "103"])
    assert r["A@4e-3-s1"][0] is None


def test_nan_counts_from_step_2000_as_registered():
    # The rule scores events after warmup (step 2,000); a nan that starts in
    # warmup and persists is caught at the first logged step after it.
    jf, d = setup({"x": [log(last=2500, gn=lambda s: float("nan") if s >= 1000 else 0.5)]})
    assert nan_runs(jf, d)[0][1] == 2020
    jf, d = setup({"x": [log(last=2500, gn=lambda s: float("nan") if s == 1000 else 0.5)]})
    assert nan_runs(jf, d)[0][1] is None


def test_spike_is_not_nan():
    jf, d = setup({"x": [log(last=5000, gn=lambda s: 50.0 if s == 4000 else 0.5)]})
    assert nan_runs(jf, d)[0][1] is None


def test_no_log_yet():
    jf, d = setup({"x": [None, None]})
    assert nan_runs(jf, d) == [("x", None, ["101", "102"])]


if __name__ == "__main__":
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
