#!/usr/bin/env python3
"""Tests for scripts/score_lr_headroom.py (CPU, no data, no sacct).

    python scripts/test_score_lr_headroom.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_lr_headroom import attribute, readings  # noqa: E402
from test_stability_flags import log  # noqa: E402

C = ("clean", 16.0, None)
F = ("failed", None, "x")
P = ("pending", None, None)


def grid(**kw):
    R = {"A@2e-3": C, "A@4e-3": C, "D@2e-3": C, "D@4e-3": C,
         "LS@3e-4": C, "LS@6e-4": ("clean", 20.0, None)}
    R.update({k.replace("_", "@"): v for k, v in kw.items()})
    return R


def has(out, text):
    return any(text in line for line in out)


def test_h1_and_trigger():
    out = readings(grid())
    assert has(out, "Q1 H1 wide")
    assert has(out, "CONDITIONAL STAGE TRIGGERED: run A@8e-3 and D@8e-3")


def test_h1_lower_bound_at_8e3():
    R = grid()
    R["A@8e-3"], R["D@8e-3"] = C, F
    out = readings(R)
    assert has(out, ">= 8x (lower bound)") and not has(out, "TRIGGERED")
    assert has(out, "Q3 @8e-3 Stieltjes is more robust")
    assert has(out, "= 0.008 / 0.0006 = 13.33x (lower bound")


def test_h2_no_trigger():
    out = readings(grid(**{"A_4e-3": F}))
    assert has(out, "Q1 H2 moderate") and not has(out, "TRIGGERED")
    assert has(out, "Q3 @4e-3 the norm is more robust: D clean, A failed")


def test_h3():
    out = readings(grid(**{"A_2e-3": F, "A_4e-3": F}))
    assert has(out, "Q1 H3 narrow")
    assert has(out, "Q2 highest clean LS lr 0.0006; Stieltjes / matched softmax"
                    " = 0.001 / 0.0006 = 1.67x")
    assert not has(out, "lower bound")


def test_pending_withheld():
    out = readings(grid(**{"A_4e-3": P}))
    assert has(out, "Q1 pending") and has(out, "Q2 pending")
    assert has(out, "Q3 @4e-3 pending") and not has(out, "TRIGGERED")


def test_q3_ratio_branches():
    R = grid()
    R["A@2e-3"], R["D@2e-3"] = ("clean", 16.4, None), ("clean", 16.0, None)
    assert has(readings(R), "Q3 @2e-3 partial: both clean, A/D = 1.025") is False
    assert has(readings(R), "Q3 @2e-3 matches the cure: both clean, A/D = 1.025")
    R["A@2e-3"] = ("clean", 16.7, None)
    assert has(readings(R), "Q3 @2e-3 partial: both clean, A/D = 1.044")
    R["A@2e-3"] = ("clean", 16.9, None)
    assert has(readings(R), "Q3 @2e-3 the norm is more robust: both clean, A/D = 1.056")


def test_q4_branches():
    R = grid()
    R["LS@6e-4"] = ("failed", 19.5, "x")       # reached 9,500 with spikes
    assert has(readings(R), "Q4 caveat removed at 6e-4")
    R["LS@6e-4"] = ("clean", 17.2, None)
    assert has(readings(R), "Q4 the learnable scale stabilizes softmax at 6e-4")
    R["LS@6e-4"] = ("clean", 18.5, None)
    assert has(readings(R), "Q4 partial")
    R["LS@6e-4"] = ("failed", None, "x")       # nan before 9,500
    assert has(readings(R), "outside the pre-registered rule")
    assert has(readings(R, {"LS@6e-4"}), "Q4 pending: LS@6e-4 failed but is still training")


def test_q2_undefined():
    out = readings(grid(**{"LS_3e-4": F, "LS_6e-4": F}))
    assert has(out, "ratio undefined")


def test_attribution():
    probe = {s: 50.0 for s in range(0, 6000, 20)}
    probe.update({s: 150.0 for s in range(4500, 6000, 20)})
    f = {"first_gn5": 5000, "first_mono": None, "first_bad": None}
    assert "-> attention-logit-driven" in attribute(f, probe)   # 150 / 50
    f = {"first_gn5": 4400, "first_mono": None, "first_bad": None}
    assert "-> not attributed" in attribute(f, probe)          # 50 / 50
    f = {"first_gn5": None, "first_mono": None, "first_bad": None}
    assert "not attributable" in attribute(f, probe)


def test_end_to_end():
    d = tempfile.mkdtemp()
    runs = {"A@2e-3": (1, 2, {}), "A@4e-3": (3, 4, {"gn": lambda s: 9.0 if s == 6000 else 0.5}),
            "D@2e-3": (5, 6, {}), "D@4e-3": (7, 8, {}),
            "LS@3e-4": (9, 10, {}), "LS@6e-4": (11, 12, {"ppl": lambda s: 21.0})}
    with open(os.path.join(d, "jobs.txt"), "w") as fh:
        for name, (j1, j2, kw) in runs.items():
            fh.write(f"{name} {j1} {j2}\n")
            os.replace(log(last=4000, **kw), os.path.join(d, f"qknorm-pilot-{j1}.out"))
            os.replace(log(**kw), os.path.join(d, f"qknorm-pilot-{j2}.out"))
    out = subprocess.run([sys.executable, os.path.join(HERE, "score_lr_headroom.py"),
                          "--jobs", os.path.join(d, "jobs.txt"), "--results", d,
                          "--assume-ended"], capture_output=True, text=True, check=True).stdout
    assert "Q1 H2 moderate" in out, out
    assert "Q4 caveat removed at 6e-4: LS@6e-4 21.00" in out, out
    assert "attribution A@4e-3: no probe before" in out, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASS")
