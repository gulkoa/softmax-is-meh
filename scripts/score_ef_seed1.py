#!/usr/bin/env python3
"""Score the seed-1 replicate of the 124M modern pair E / F, pre-registered in
thesis/experiments/2026-10-18-ef-seed1-124m.md (login node).

In distribution: the step-9,500 validation losses (4 decimals) of the merged
chunk logs (results/backbone-pilot-<job>.out, chunks from
results/ef_seed1_jobs.txt; on a re-run step the later chunk wins). Each arm
is checked against its seed 0 (+-0.5 %), and the pair gap relative to
softmax against the 0.5 % tie bar and the 1 % seed-effect bar, in the
wording fixed before submission.

Length: the seed-1 uniform-grid eval log (--length; --ext adds the
pre-authorised edge extension). Its seed-0 blocks must reproduce 15637264
on the columns off, 0.25, 0.3, 0.35 and 0.5 at 2x / 4x / 8x, or nothing is
scored. Then, per length:
  - each seed-1 arm's best deep-tail perplexity over its grid (as printed,
    'off' excluded) and the edge flag;
  - the ratio F / E with score_length_alpha_grid.verdict's wording;
  - the joint pattern with seed 0, recomputed from 15637264.

    python scripts/score_ef_seed1.py [--length LOG] [--ext LOG] [--results DIR]
"""
import argparse
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_length_alpha_grid import LENGTHS, best, gate, parse_blocks, union, verdict  # noqa: E402

E0 = "ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt"
F0 = "ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt"
E1 = E0.replace("_s0.pt", "_s1.pt")
F1 = F0.replace("_s0.pt", "_s1.pt")
L0 = {"E": 2.7710, "F": 2.7708}   # seed 0, step-9,500 validation losses
REF = "length-alpha-uniform-124m-ef-15637264.out"
GATE_COLS = (0.25, 0.3, 0.35, 0.5)
LO, HI = 0.05, 0.5
VAL = re.compile(r"^\s*\[val\] step (\d+) loss (\S+) ppl", re.M)


def val_losses(paths):
    """{step: loss} from chunk logs in chunk order; a re-run step keeps the later chunk's."""
    out = {}
    for p in paths:
        if os.path.exists(p):
            out.update({int(s): float(v) for s, v in VAL.findall(open(p, errors="ignore").read())})
    return out


def pct(dlog):
    """exp(dlog) - 1, in percent."""
    return 100 * (math.exp(dlog) - 1)


def id_reading(le, lf):
    """The pre-registered in-distribution lines from the step-9,500 losses of E-s1 and F-s1."""
    de, df, gap = pct(le - L0["E"]), pct(lf - L0["F"]), pct(le - lf)
    ok = {"E": abs(de) <= 0.5, "F": abs(df) <= 0.5}
    lines = [f"E-s1 vs seed 0: {de:+.2f} % ({'inside' if ok['E'] else 'outside'} +-0.5 %)",
             f"F-s1 vs seed 0: {df:+.2f} % ({'inside' if ok['F'] else 'outside'} +-0.5 %)",
             f"pair gap relative to softmax: {gap:+.2f} %"]
    if abs(gap) > 1:
        lines.append("a seed effect to report, not a new ranking")
    elif abs(gap) > 0.5:
        lines.append(f"the tie does not replicate at the pre-registered 0.5 % bar (gap {gap:+.2f} %); "
                     "it is below the 1 % seed-effect bar.")
    elif all(ok.values()):
        lines.append("the tie replicates")
    else:
        lines.append("the gap is inside the 0.5 % bar but a seed-0 check fails: "
                     "each check as it falls, without \"the tie replicates\"")
    return lines


def band(r):
    return "A" if r >= 1.08 else "B" if r >= 1.0 else "C"


def joint(r0, r1, k):
    """The pre-registered joint wording for seed 0's and seed 1's ratios at k x."""
    b = {band(r0), band(r1)}
    if b == {"A"}:
        return f"the {k}x lead replicates at the 1.08 margin (2 of 2 seeds)"
    if b == {"A", "B"}:
        return f"Stieltjes is ahead at {k}x at both seeds, at the 1.08 margin at one of two"
    if b == {"B"}:
        return f"Stieltjes is ahead at {k}x at both seeds, by less than the 1.08 margin at both"
    if b == {"C"}:
        return f"softmax is ahead at {k}x at both seeds"
    return f"the seeds disagree at {k}x, and no ranking is claimed"


def length_lines(ref, new, ext=None):
    """Score the seed-1 blocks of `new` (parse_blocks output) against seed 0 in `ref`."""
    ext = ext or {}
    ok = {"E-s0": gate(new[E0], ref[E0], GATE_COLS), "F-s0": gate(new[F0], ref[F0], GATE_COLS)}
    lines = [f"reproduction gates (seed 0 vs 15637264): {ok}"]
    if not all(ok.values()):
        return lines + ["gate failed: the seed-1 blocks are not scored"]
    for L in LENGTHS:
        k = L // 1024
        r0 = best(union(ref[F0][L]), LO, HI)[0] / best(union(ref[E0][L]), LO, HI)[0]
        res = {}
        for arm, n in (("E", E1), ("F", F1)):
            row = union(new[n][L], ext.get(n, {}).get(L, {}))
            lo, hi = (min(row), max(row)) if n in ext else (LO, HI)
            v, al, edge = best(row, lo, hi)
            res[arm] = (v, edge, n in ext)
            lines.append(f"{k}x {arm}-s1: best {v} at alpha {al}{'  EDGE' if edge else ''}"
                         f"  (off {new[n][L]['off']})")
        (ve, ee, xe), (vf, ef, xf) = res["E"], res["F"]
        r = vf / ve
        if ee or ef:
            if (ee and not xe) or (ef and not xf):
                lines.append(f"{k}x: edge-limited before the extension; run the pre-authorised "
                             "extension for the edge-limited arm")
            else:
                bound = ">=" if ee and not ef else "<=" if ef and not ee else "unbounded:"
                lines.append(f"{k}x: edge-limited after the extension: ratio {bound} {r:.3f} "
                             "(no verdict phrase)")
            continue
        lines.append(f"{k}x: ratio F/E {vf} / {ve} = {r:.3f} | {verdict(r, k)}")
        lines.append(f"{k}x: seed 0 ratio {r0:.3f} | joint: {joint(r0, r, k)}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results"))
    ap.add_argument("--length", help="the seed-1 uniform-grid eval log")
    ap.add_argument("--ext", help="the pre-authorised edge-extension eval log")
    a = ap.parse_args()
    jobs = {}
    for line in open(os.path.join(a.results, "ef_seed1_jobs.txt")):
        if line.strip():
            name, *ids = line.split()
            jobs[name] = [os.path.join(a.results, f"backbone-pilot-{j}.out") for j in ids]
    le, lf = (val_losses(jobs[n]).get(9500) for n in ("E-s1", "F-s1"))
    if le is None or lf is None:
        print(f"in distribution: no step-9,500 validation yet (E-s1 {le}, F-s1 {lf})")
    else:
        print(f"step-9,500 validation losses: E-s1 {le:.4f}, F-s1 {lf:.4f}")
        for line in id_reading(le, lf):
            print(" ", line)
    if a.length:
        ref = parse_blocks(os.path.join(a.results, REF))
        ext = parse_blocks(a.ext) if a.ext else None
        for line in length_lines(ref, parse_blocks(a.length), ext):
            print(line)


if __name__ == "__main__":
    main()
