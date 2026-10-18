#!/usr/bin/env python3
"""Score the uniform alpha-grid length evals of
thesis/experiments/2026-10-17-length-alpha-uniform-grid.md (login node).

Reads eval_longctx_scaled.py logs ("=== <ckpt>" blocks, or one unnamed
block), checks the reproduction gates (the repeated columns must match the
earlier evals cell for cell), takes each arm's best deep-tail perplexity
over the union of its sweeps (as printed, no interpolation), flags an
optimum that sits only at an end of the arm's uniform range, and prints
the softmax / Stieltjes ratios in the wording fixed before submission.

    python scripts/score_length_alpha_grid.py [--results DIR]
"""
import argparse
import os
import re

LENGTHS = (2048, 4096, 8192)
P2 = {"S1": "ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s1.pt",
      "T1": "ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s1.pt",
      "S0": "ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s0.pt",
      "T0": "ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s0.pt"}
N0 = "ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0.fromhf.pt"
E = "ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt"
F = "ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt"
RANGES = {"S1": (0.15, 0.5), "T1": (0.15, 0.5), "S0": (0.15, 0.5),
          "T0": (0.15, 0.5), "N0": (0.225, 0.5), "E": (0.05, 0.5),
          "F": (0.05, 0.5)}
COMPARISONS = (("matched pair, s1", "T1", "S1"), ("matched pair, s0", "T0", "S0"),
               ("N0 vs softmax + QK-norm s0", "T0", "N0"), ("F vs E", "F", "E"))


def parse_blocks(path):
    """{ckpt name or None: {L: {alpha or 'off': ppl}}} from one eval log."""
    out, name, alphas = {}, None, None
    for line in open(path):
        m = re.match(r"=== (\S+)", line)
        if m:
            name = m.group(1)
            out[name] = {}
            continue
        m = re.search(r"alpha sweep=\[([^\]]*)\]", line)
        if m:
            alphas = [float(x) for x in m.group(1).split(",")]
            out.setdefault(name, {})
            continue
        f = line.split()
        if alphas and f and f[0] in ("1024", "2048", "4096", "8192"):
            out.setdefault(name, {})[int(f[0])] = dict(
                zip(["off"] + alphas, [float(x) for x in f[1:]]))
    return out


def gate(new, old, cols):
    """True when the off column and every gate column match at 2x, 4x, 8x."""
    return all(new[L][c] == old[L][c] for L in LENGTHS for c in ["off"] + list(cols))


def union(*rows):
    """Merge alpha -> ppl rows (later rows win on a repeated alpha; 'off' dropped)."""
    u = {}
    for r in rows:
        u.update({a: v for a, v in r.items() if a != "off"})
    return u


def best(row, lo, hi):
    """(ppl, alphas at the minimum, edge): edge when every minimising alpha is a range end."""
    v = min(row.values())
    al = sorted(a for a, x in row.items() if x == v)
    ends = [a for a in al if abs(a - lo) < 1e-9 or abs(a - hi) < 1e-9]
    return v, al, len(ends) == len(al)


def verdict(r, k):
    """The wording fixed before submission, per comparison and length k."""
    if r >= 1.08:
        return f"on the uniform grid, Stieltjes is ahead at {k}x by at least the 1.08 margin (ratio {r:.3f})"
    if r >= 1.0:
        return f"on the uniform grid, Stieltjes is ahead at {k}x by less than the 1.08 margin (ratio {r:.3f})"
    return f"on the uniform grid, softmax is ahead at {k}x (ratio {r:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    R = ap.parse_args().results
    rd = lambda f: parse_blocks(os.path.join(R, f))  # noqa: E731
    proto, edge = rd("p2s1-length-evals-15534295.out"), rd("p2-length-alpha-edge-15637051.out")
    uni, ef = rd("length-alpha-uniform-355m-15637263.out"), rd("length-alpha-uniform-124m-ef-15637264.out")
    n0old = rd("code-rl-stage-13297095.out")[None]
    eold, fold = rd("code-rl-stage-14072702.out")[None], rd("code-rl-stage-14072703.out")[None]
    gates = {k: gate(uni[n], proto[n], [0.35]) for k, n in P2.items()}
    gates.update(N0=gate(uni[N0], n0old, [0.35]),
                 E=gate(ef[E], eold, [0.25, 0.3, 0.35, 0.5]),
                 F=gate(ef[F], fold, [0.25, 0.3, 0.35, 0.5]))
    print("reproduction gates:", gates)
    grids = {k: {L: union(proto[n][L], edge[n][L], uni[n][L]) for L in LENGTHS} for k, n in P2.items()}
    grids["N0"] = {L: union(n0old[L], uni[N0][L]) for L in LENGTHS}
    grids["E"] = {L: union(ef[E][L]) for L in LENGTHS}
    grids["F"] = {L: union(ef[F][L]) for L in LENGTHS}
    b = {}
    for k, g in grids.items():
        if not gates[k]:
            print(f"{k}: gate failed, not scored")
            continue
        for L in LENGTHS:
            v, al, e = best(g[L], *RANGES[k])
            b[k, L] = v
            print(f"{k} {L}: best {v} at alpha {al}{'  EDGE' if e else ''}")
    for name, t, s in COMPARISONS:
        for L in LENGTHS:
            if (t, L) in b and (s, L) in b:
                r = b[t, L] / b[s, L]
                print(f"{name} {L // 1024}x: {b[t, L]} / {b[s, L]} = {r:.3f} | {verdict(r, L // 1024)}")


if __name__ == "__main__":
    main()
