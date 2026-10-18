#!/usr/bin/env python3
"""Score the uniform alpha-grid bits-per-byte length eval of the 124M
tokenizer cell (thesis/experiments/2026-10-18-len-bpb-alpha-uniform-124m.md;
login node).

The tok32k eval prints, per "=== <ckpt>" block, a deep-tail perplexity
table and then a bits-per-byte table on the same text. Gates: the off,
0.25, 0.3, 0.35 and 0.5 columns of both tables must match job 15511190.
Each arm's best bits per byte over the grid (as printed, no
interpolation; an optimum at a grid end is an edge), then the ratios
softmax / Stieltjes at 4x and 8x against 1.05 and 1.0, in the wording
fixed before submission.

    python scripts/score_len_bpb_alpha_grid.py JOB [--results DIR]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_length_alpha_grid import best  # noqa: E402

LENGTHS = (2048, 4096, 8192)
CK = {"E": "ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt",
      "F": "ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt",
      "E'": "ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt",
      "F'": "ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt"}
GATE_COLS = ["off", 0.25, 0.3, 0.35, 0.5]


def parse_two_tables(path):
    """{ckpt: {'ppl': {L: {alpha|'off': v}}, 'bpb': {...}}}: the first table
    of a block is perplexity, the one after the 'bits per byte' line is bpb."""
    out, name, alphas, kind = {}, None, None, "ppl"
    for line in open(path):
        m = re.match(r"=== (\S+)", line)
        if m:
            name, kind = m.group(1), "ppl"
            out[name] = {"ppl": {}, "bpb": {}}
            continue
        m = re.search(r"alpha sweep=\[([^\]]*)\]", line)
        if m:
            alphas = [float(x) for x in m.group(1).split(",")]
            continue
        if "bits per byte" in line:
            kind = "bpb"
            continue
        f = line.split()
        if name and alphas and f and f[0] in ("1024", "2048", "4096", "8192"):
            out[name][kind][int(f[0])] = dict(zip(["off"] + alphas, [float(x) for x in f[1:]]))
    return out


def verdict(r, k):
    if r >= 1.05:
        return f"on the uniform grid, Stieltjes is ahead at {k}x by at least the 1.05 margin (bits-per-byte ratio {r:.3f})"
    if r >= 1.0:
        return f"on the uniform grid, Stieltjes is ahead at {k}x by less than the 1.05 margin (ratio {r:.3f})"
    return f"on the uniform grid, softmax is ahead at {k}x (ratio {r:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    a = ap.parse_args()
    new = parse_two_tables(os.path.join(a.results, f"len-bpb-alpha-uniform-{a.job}.out"))
    old = parse_two_tables(os.path.join(a.results, "len-bpb-15511190.out"))
    b = {}
    for k, n in CK.items():
        ok = all(new[n][t][L][c] == old[n][t][L][c] for t in ("ppl", "bpb") for L in LENGTHS for c in GATE_COLS)
        print(f"{k}: gate {'pass' if ok else 'FAIL (not scored)'}")
        if not ok:
            continue
        for L in LENGTHS:
            row = {al: v for al, v in new[n]["bpb"][L].items() if al != "off"}
            v, al, e = best(row, min(row), max(row))
            pv, pal, _ = best({x: y for x, y in new[n]["ppl"][L].items() if x != "off"}, min(row), max(row))
            b[k, L] = v
            print(f"  {k} {L}: best bpb {v} at alpha {al}{'  EDGE' if e else ''} | best ppl {pv} at {pal}")
    for t, s in (("F", "E"), ("F'", "E'")):
        for L in LENGTHS:
            if (t, L) in b and (s, L) in b:
                r = b[t, L] / b[s, L]
                print(f"{t}/{s} {L // 1024}x: {b[t, L]} / {b[s, L]} = {r:.3f} | {verdict(r, L // 1024)}")
    for x, y in (("E'", "E"), ("F'", "F")):
        if (x, 8192) in b and (y, 8192) in b:
            print(f"tokenizer change at 8x, {x} vs {y}: {100 * (b[x, 8192] / b[y, 8192] - 1):+.1f} % bits")


if __name__ == "__main__":
    main()
