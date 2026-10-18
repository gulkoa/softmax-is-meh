#!/usr/bin/env python3
"""Tests for scripts/score_ef_seed1.py (CPU, synthetic logs; one test also
reads the committed seed-0 eval log when it is present).

    python scripts/test_score_ef_seed1.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_ef_seed1 import E0, E1, F0, F1, REF, id_reading, joint, length_lines, val_losses  # noqa: E402
from score_length_alpha_grid import LENGTHS, parse_blocks  # noqa: E402

ALPHAS = [round(0.05 + 0.025 * i, 3) for i in range(19)]


def grid(opt, scale, alphas=ALPHAS):
    """{L: {'off': .., alpha: ..}}: a bowl with its minimum at `opt`, times `scale`."""
    return {L: {"off": 1000.0, **{a: round(scale * (100 + 1000 * (a - opt) ** 2), 1) for a in alphas}}
            for L in (1024,) + LENGTHS}


def test_id_reading_bands():
    assert id_reading(2.7710, 2.7708)[-1] == "the tie replicates"
    assert id_reading(2.7749, 2.7700)[-1] == "the tie replicates"             # gap +0.49 %
    assert id_reading(2.7750, 2.7700)[-1].startswith("the tie does not replicate")  # +0.50 %
    assert id_reading(2.7800, 2.7700)[-1] == "a seed effect to report, not a new ranking"  # +1.005 %
    assert id_reading(2.7649, 2.7750)[-1] == "a seed effect to report, not a new ranking"  # -1.005 %
    assert id_reading(2.7650, 2.7750)[-1].startswith("the tie does not replicate")  # -0.995 %
    # gap 0, but E is +0.50 % from its seed 0 (outside +-0.5 %)
    r = id_reading(2.7760, 2.7760)
    assert "outside" in r[0] and r[-1].startswith("the gap is inside the 0.5 % bar but a seed-0 check fails")
    assert id_reading(2.7660, 2.7708)[0].endswith("(inside +-0.5 %)")       # -0.4988 %
    assert id_reading(2.7659, 2.7708)[0].endswith("(outside +-0.5 %)")      # -0.5087 %


def test_joint_wording():
    assert joint(1.475, 1.20, 8) == "the 8x lead replicates at the 1.08 margin (2 of 2 seeds)"
    assert joint(1.475, 1.05, 8) == "Stieltjes is ahead at 8x at both seeds, at the 1.08 margin at one of two"
    assert joint(1.063, 1.0, 4) == "Stieltjes is ahead at 4x at both seeds, by less than the 1.08 margin at both"
    assert joint(0.97, 0.99, 2) == "softmax is ahead at 2x at both seeds"
    assert joint(1.010, 0.999, 2) == "the seeds disagree at 2x, and no ranking is claimed"
    assert joint(1.475, 0.95, 8) == "the seeds disagree at 8x, and no ranking is claimed"


def test_val_losses_later_chunk_wins():
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, "a.out"), os.path.join(d, "b.out")
    open(a, "w").write("  [val] step 9250 loss 2.7800 ppl 16.12 (wall)\n"
                       "  [val] step 9500 loss 2.7790 ppl 16.10 (wall)\n")
    open(b, "w").write("  [val] step 9500 loss 2.7712 ppl 15.98 (wall)\n")
    v = val_losses([a, b, os.path.join(d, "missing.out")])
    assert v == {9250: 2.7800, 9500: 2.7712}


def test_length_lines_scores_and_gates():
    ref = {E0: grid(0.375, 1.0), F0: grid(0.2, 1.3)}
    new = {E0: ref[E0], F0: ref[F0], E1: grid(0.3, 1.0), F1: grid(0.2, 1.1)}
    out = "\n".join(length_lines(ref, new))
    assert "{'E-s0': True, 'F-s0': True}" in out
    assert "8x: ratio F/E 110.0 / 100.0 = 1.100 | on the uniform grid, Stieltjes is ahead at 8x " \
           "by at least the 1.08 margin (ratio 1.100)" in out
    assert "8x: seed 0 ratio 1.300 | joint: the 8x lead replicates at the 1.08 margin (2 of 2 seeds)" in out
    bad = dict(new)
    bad[E0] = {L: dict(r) for L, r in new[E0].items()}
    bad[E0][8192][0.35] += 0.1
    assert length_lines(ref, bad)[-1] == "gate failed: the seed-1 blocks are not scored"


def test_edge_then_extension():
    ref = {E0: grid(0.375, 1.0), F0: grid(0.2, 1.3)}
    new = {E0: ref[E0], F0: ref[F0], E1: grid(0.6, 1.0), F1: grid(0.2, 1.1)}   # E-s1 optimum above 0.5
    out = length_lines(ref, new)
    assert any("8x E-s1: best" in x and "EDGE" in x for x in out)
    assert any(x.startswith("8x: edge-limited before the extension") for x in out)
    ext_alphas = [round(0.525 + 0.025 * i, 3) for i in range(8)]                # 0.525-0.70
    ext = {E1: grid(0.6, 1.0, ext_alphas)}
    out = "\n".join(length_lines(ref, new, ext))
    assert "8x E-s1: best 100.0 at alpha [0.6]" in out and "edge-limited" not in out
    ext = {E1: grid(0.9, 1.0, ext_alphas)}                                       # still at the end
    out = "\n".join(length_lines(ref, {**new, E1: grid(0.9, 1.0)}, ext))
    assert "8x: edge-limited after the extension: ratio >= " in out


def test_seed0_copy_of_the_real_log_reproduces_the_registered_ratios():
    p = os.path.join(os.path.dirname(HERE), "results", REF)
    if not os.path.exists(p):
        print("  (skipped: no", REF, ")")
        return
    ref = parse_blocks(p)
    new = {E0: ref[E0], F0: ref[F0], E1: ref[E0], F1: ref[F0]}
    out = "\n".join(length_lines(ref, new))
    for k, r in ((2, "1.010"), (4, "1.063"), (8, "1.475")):
        assert f"{k}x: seed 0 ratio {r}" in out, (k, r)
    assert "joint: the 8x lead replicates at the 1.08 margin (2 of 2 seeds)" in out
    assert "joint: Stieltjes is ahead at 4x at both seeds, by less than the 1.08 margin at both" in out


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
    print("ALL OK")
