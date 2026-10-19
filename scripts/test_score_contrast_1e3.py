#!/usr/bin/env python3
"""Tests for scripts/score_contrast_1e3.py (CPU, synthetic logs, no sacct).

    python scripts/test_score_contrast_1e3.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_contrast_1e3 import (arm_phrase, collapse_readings, contrast_phrase,  # noqa: E402
                                pair, parse_signature, ratio_phrase, sign_test, status)
from test_stability_flags import log  # noqa: E402


def test_status():
    assert status("FAILED", False) == "failed"          # one event is final
    assert status("CLEAN", True) == "clean"
    assert status("INCOMPLETE", False) == "pending"
    assert status("INCOMPLETE", True) == "unresolved"
    assert status("INCOMPLETE", True, "training") == "failed"
    assert status("INCOMPLETE", True, "operational") == "no verdict"


def test_pair():
    assert pair("clean", "failed") == "holds"
    assert pair("failed", "clean") == "reversed"
    assert pair("clean", "clean") == "no contrast (both clean)"
    assert pair("failed", "failed") == "no contrast (both failed)"
    assert pair("clean", "no verdict") == "no verdict"
    assert pair("pending", "failed") == "pending"
    assert pair("clean", "unresolved") == "pending"


def test_arm_phrase():
    a = {0: "clean", 1: "clean", 2: "clean", 3: "clean", 4: "clean"}
    assert arm_phrase("A", a) == "A: 0 of 3 further seeds failed at 1e-3 (0 of 5 with seeds 0 and 1)"
    ls = {0: "failed", 1: "failed", 2: "failed", 3: "clean", 4: "failed"}
    assert arm_phrase("LS", ls) == "LS: 2 of 3 further seeds failed at 1e-3 (4 of 5 with seeds 0 and 1)"
    ls[4] = "no verdict"
    assert arm_phrase("LS", ls) == "LS: 1 of 2 further seeds failed at 1e-3 (3 of 4 with seeds 0 and 1)"
    ls[4] = "pending"
    assert arm_phrase("LS", ls) == "LS: pending"


def test_contrast_phrase():
    p = {0: "holds", 1: "holds", 2: "holds", 3: "holds", 4: "holds"}
    assert contrast_phrase(p) == ("the contrast holds at 3 of 3 further seeds "
                                  "(5 of 5 with seeds 0 and 1)")
    p[3], p[4] = "reversed", "no verdict"
    assert contrast_phrase(p) == ("the contrast holds at 1 of 2 further seeds "
                                  "(3 of 4 with seeds 0 and 1); s3 reversed, s4 no verdict")
    p[2] = "pending"
    assert contrast_phrase(p) == "contrast: pending"


def test_ratio_phrase():
    r = {0: 1.9, 1: 2.1, 2: 1.15, 3: 1.149, 4: None}
    assert ratio_phrase(r, (2, 3, 4)) == "ratio ≥ 1.15 at 1 of 2 seeds (undefined: s4)"
    assert ratio_phrase(r, (0, 1, 2, 3, 4)) == "ratio ≥ 1.15 at 3 of 4 seeds (undefined: s4)"
    r[3] = "pending"
    assert ratio_phrase(r, (2, 3, 4)) == "ratio: pending"


def test_sign_test():
    assert sign_test(5, 0) == 0.03125
    assert sign_test(4, 0) == 0.0625
    assert sign_test(3, 1) == 0.3125
    assert sign_test(0, 0) is None


def sig_file(lines):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for ck, step, total in lines:
        f.write(f"SIGNATURE {ck} step {step}: collapsed heads per layer "
                f"[0, 0, {total}, 0], total {total}; shape-only {total} [0, 0, {total}, 0]\n")
    f.close()
    return f.name


def test_parse_signature_and_tallies():
    ls, a = "ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s", "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s"
    lines = [(f"{a}{s}.pt", 9535, 0) for s in (2, 3, 4)]
    lines += [(f"{ls}2.step4000.pt", 4000, 0), (f"{ls}2.pt", 9535, 0),
              (f"{ls}3.step4000.pt", 4000, 2), (f"{ls}3.pt", 9535, 12),
              (f"{ls}4.step5000.pt", 5000, 0), (f"/x/{ls}4.pt", 9535, 0),
              ("ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0003_s3.pt", 9535, 99)]
    sig = parse_signature(sig_file(lines))
    assert sig["LS@1e-3-s3"]["snap"] == {4000: 2}
    assert sig["LS@1e-3-s3"]["final"] == (9535, 12, 12)       # the 3e-4 line is ignored
    assert sig["LS@1e-3-s4"]["final"] == (9535, 0, 0)          # path prefix stripped
    st = {"A@1e-3-s2": "clean", "A@1e-3-s3": "clean", "A@1e-3-s4": "clean",
          "LS@1e-3-s2": "clean", "LS@1e-3-s3": "failed", "LS@1e-3-s4": "failed"}
    gn5 = {"LS@1e-3-s3": 6000, "LS@1e-3-s4": None}
    out = collapse_readings(st, gn5, sig)
    assert "1 of the 2 failed LS runs end with collapsed heads, and 1 of the 1 clean LS runs have none" in out
    assert "0 of 3 A runs end with collapsed heads" in out
    assert "exception: LS@1e-3-s4 failed without collapsed heads" in out
    assert "in 1 of the 2 failed LS runs a snapshot shows collapse before the first gradient norm above 5" in out
    assert "  LS@1e-3-s4: no gradient norm above 5" in out
    st["LS@1e-3-s2"] = "pending"
    assert collapse_readings(st, gn5, sig)[-1] == "collapse: pending"


def test_end_to_end():
    d = tempfile.mkdtemp()
    spike = {"gn": lambda s: 9.0 if s >= 5000 else 0.5, "ppl": lambda s: 30.0 - s / 1000}
    ref = {"14010298": {"last": 4000}, "14010299": {}, "15516255": {"last": 4000},
           "15516256": {}, "15530796": spike, "15530798": spike}
    for j, kw in ref.items():
        os.replace(log(**kw), os.path.join(d, f"qknorm-pilot-{j}.out"))
    probe = lambda s: 100.0 if s < 4600 else 400.0          # noqa: E731
    jobs = {"A@1e-3-s2": (1, {}), "A@1e-3-s3": (2, {}), "A@1e-3-s4": (3, {}),
            "LS@1e-3-s2": (4, dict(spike, probe=probe)), "LS@1e-3-s3": (5, spike),
            "LS@1e-3-s4": (6, {"last": 7000})}
    with open(os.path.join(d, "jobs.txt"), "w") as fh:
        for n, (j, kw) in jobs.items():
            fh.write(f"{n} {j} {j + 100}\n")              # the spare never ran
            os.replace(log(**kw), os.path.join(d, f"qknorm-pilot-{j}.out"))
    cmd = [sys.executable, os.path.join(HERE, "score_contrast_1e3.py"), "--jobs",
           os.path.join(d, "jobs.txt"), "--results", d, "--assume-ended"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert "LS: pending" in out and "contrast: pending" in out, out    # s4 unresolved
    assert "MISMATCH" not in out, out
    out = subprocess.run(cmd + ["--stopped", "LS@1e-3-s4=operational"],
                         capture_output=True, text=True, check=True).stdout
    assert "A: 0 of 3 further seeds failed at 1e-3 (0 of 5 with seeds 0 and 1)" in out, out
    assert "LS: 2 of 2 further seeds failed at 1e-3 (4 of 4 with seeds 0 and 1)" in out, out
    assert ("the contrast holds at 2 of 2 further seeds (4 of 4 with seeds 0 and 1); "
            "s4 no verdict") in out, out
    assert "seeds 2-4: ratio ≥ 1.15 at 0 of 2 seeds (undefined: s4)" in out, out
    assert "4 holds, 0 reversed, p = 0.0625" in out, out
    assert ("attribution LS@1e-3-s2: first event step 5000; M(4980) 400.0 / "
            "M(3980) 100.0 = 4.00 -> attention-logit-driven") in out, out
    assert "attribution LS@1e-3-s3: no probe before the first event" in out, out
    assert "LS@1e-3-s2: final 20.50 @9,500" in out, out
    assert "LS@1e-3-s4: final no step-9,500 validation (last 23.00 @7,000)" in out, out
    out = subprocess.run(cmd + ["--stopped", "LS@1e-3-s4=training"],
                         capture_output=True, text=True, check=True).stdout
    assert "LS: 3 of 3 further seeds failed at 1e-3 (5 of 5 with seeds 0 and 1)" in out, out
    assert "the contrast holds at 3 of 3 further seeds (5 of 5 with seeds 0 and 1)" in out, out
    assert "5 holds, 0 reversed, p = 0.03125" in out, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASS")
