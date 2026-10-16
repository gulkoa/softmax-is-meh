#!/usr/bin/env python3
"""Tests for scripts/score_logit_probe.py on synthetic logs (CPU, no data).

    python scripts/test_score_logit_probe.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_logit_probe import score  # noqa: E402


def log(last, M, gn=lambda s: 1.0, stopped=False, skip=()):
    lines = []
    for s in range(0, last + 1, 20):
        lines.append(f"step {s:6d}/9536 loss 3.1000 gnorm {gn(s):.2f} (200k tok/s)")
        if s not in skip:
            m = M(s)
            lines.append(f"probe step {s} M {m:.3f} maxabs 1.00 {m:.2f} | minent 6.000 0.500")
    if stopped:
        lines.append(f"STOPPED at step {last} (--stop-step): ckpt.pt")
    f = tempfile.NamedTemporaryFile("w", suffix=".out", delete=False)
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


SOFT = lambda s: 100.0 if s <= 2000 else 100.0 * 1.01 ** (s - 2000)   # noqa: E731
FLAT = lambda s: 80.0                                                   # noqa: E731


def test_reading1():
    sm = log(4600, SOFT, gn=lambda s: 9.0 if s == 3700 else 1.0, stopped=True)
    r, out = score(sm, log(4600, FLAT, stopped=True))
    assert r == 1, out
    assert any("M_soft(2000) 100.000 -> M_soft(3680)" in l for l in out), out


def test_timeout_without_spike_is_incomplete():
    r, out = score(log(3300, SOFT), log(4600, FLAT, stopped=True))
    assert r == "INCOMPLETE", out


def test_stopped_without_spike_is_reading4():
    r, _ = score(log(4600, SOFT, stopped=True), log(4600, FLAT, stopped=True))
    assert r == 4


def test_inf_gnorm_is_tstar():
    sm = log(4600, SOFT, gn=lambda s: float("inf") if s == 3500 else 1.0, stopped=True)
    r, out = score(sm, log(4600, FLAT, stopped=True))
    assert any("after 2000: 3500;" in l for l in out), out


def test_nan_gnorm_is_tstar():
    sm = log(4600, SOFT, gn=lambda s: float("nan") if s >= 3400 else 1.0, stopped=True)
    r, out = score(sm, log(4600, FLAT, stopped=True))
    assert any("after 2000: 3400;" in l for l in out), out


def test_chunks_later_wins():
    a = log(3300, SOFT)                                   # timed out, no spike
    b = log(4600, SOFT, gn=lambda s: 9.0 if s == 3720 else 1.0, stopped=True)
    r, out = score(f"{a},{b}", log(4600, FLAT, stopped=True))
    assert r == 1 and any("after 2000: 3720;" in l for l in out), out


def test_at_the_bar_flag():
    stj = lambda s: 80.0 if s != 3000 else 120.4                    # noqa: E731
    sm = log(4600, SOFT, gn=lambda s: 9.0 if s == 3700 else 1.0, stopped=True)
    r, out = score(sm, log(4600, stj, stopped=True))
    assert r == 5 and any("at the bar: 1.505 vs 1.5" in l for l in out), out


def test_missing_pre_probe_uses_last_before():
    sm = log(4600, SOFT, gn=lambda s: 9.0 if s == 3700 else 1.0, stopped=True, skip={3680})
    r, out = score(sm, log(4600, FLAT, stopped=True))
    assert r == 1 and any("using the last probe before t*, 3660" in l for l in out), out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASS")
