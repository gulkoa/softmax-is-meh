#!/usr/bin/env python3
"""Tests for scripts/stability_flags.py on synthetic logs (CPU, no data).

    python scripts/test_stability_flags.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stability_flags import flags, overwritten, parse  # noqa: E402


def log(last=9535, gn=lambda s: 0.5, ppl=lambda s: 30.0 - s / 1000,
        skip=(), probe=None):
    lines = []
    for s in range(0, last + 1):
        if s % 20 == 0 and s not in skip:
            g = gn(s)
            loss = "nan" if g != g else "3.1"
            lines.append(f"step {s:6d}/9536 loss {loss} gnorm {g:.2f} (200k tok/s)")
            if probe is not None:
                lines.append(f"probe step {s} M {probe(s):.3f} maxabs 1.0 1.0 | "
                             "minent 6.5 6.5")
        if s % 250 == 0 and s > 0:
            lines.append(f"  [val] step {s} loss 3.0000 ppl {ppl(s):.2f} (wall 1h)")
    f = tempfile.NamedTemporaryFile("w", suffix=".out", delete=False)
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


def run(path, start=2000, final=9500):
    return flags(*parse(path), start, final)


def test_clean():
    assert run(log())["verdict"] == "CLEAN"


def test_spike_fails():
    f = run(log(gn=lambda s: 7.0 if s == 4000 else 0.5))
    assert f["verdict"] == "FAILED" and f["first_gn5"] == 4000 and f["n_gn5"] == 1


def test_spike_in_warmup_ignored():
    assert run(log(gn=lambda s: 7.0 if s == 1000 else 0.5))["verdict"] == "CLEAN"


def test_nan_gnorm_fails():
    f = run(log(gn=lambda s: float("nan") if s >= 5000 else 0.5))
    assert f["verdict"] == "FAILED" and f["first_bad"] == 5000
    assert f["max_gn"] == 0.5


def test_inf_val_fails():
    f = run(log(ppl=lambda s: float("inf") if s == 6000 else 20.0))
    assert f["verdict"] == "FAILED" and f["first_bad"] == 6000


def test_val_jump_fails():
    f = run(log(ppl=lambda s: 25.0 if s == 6000 else 20.0))
    assert f["verdict"] == "FAILED" and f["first_mono"] == 6000


def test_incomplete():
    f = run(log(last=6000))
    assert f["verdict"] == "INCOMPLETE" and f["final_step"] == 6000


def test_missing_lines_counted():
    assert run(log(skip={3000, 3020}))["missing"] == 2


def test_chunks_later_wins():
    a = log(last=5000, gn=lambda s: 9.0 if s == 4600 else 0.5)
    b = log()                       # the resumed chunk re-ran 4,501+ cleanly
    assert run(f"{a},{b}")["verdict"] == "CLEAN"


def test_overwritten_listed_not_counted():
    a = log(last=5000, gn=lambda s: 9.0 if s == 4600 else 0.5)
    b = log()                       # re-ran 4,501+ without the spike
    assert run(f"{a},{b}")["verdict"] == "CLEAN"
    assert overwritten(f"{a},{b}", 2000) == [(4600, "gnorm>5")]


def test_overwritten_nan_and_kept_events():
    a = log(last=5000, gn=lambda s: float("nan") if s == 4800 else
            (8.0 if s == 3000 else 0.5))
    b = log(gn=lambda s: 8.0 if s == 3000 else 0.5)   # re-runs keep 3,000
    assert run(f"{a},{b}")["first_gn5"] == 3000
    assert overwritten(f"{a},{b}", 2000) == [(4800, "nan/inf")]


def test_onset_first_probe_above_1000():
    f = run(log(probe=lambda s: 5.0 + s if s < 3000 else 2000.0))
    assert f["M_onset"] == 1000      # probes every 20 steps; 5 + 1000 > 1000 first
    assert f["M_start"] == 2005.0
    assert run(log(probe=lambda s: 50.0))["M_onset"] is None
    assert run(log())["M_onset"] is None                      # no probe lines


def test_no_overwrite_single_chunk():
    assert overwritten(log(gn=lambda s: 9.0 if s == 4000 else 0.5), 2000) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASS")
