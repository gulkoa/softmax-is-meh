#!/usr/bin/env python3
"""Stability flags for 124M pilot runs, from the training logs.

A run is CLEAN when, after step --from (default 2,000, the end of warmup):
  - no logged gradient norm exceeds 5 (logged every 20 steps, clip 1.0), and
  - no validation perplexity exceeds 1.05 x the previous validation.
These are the rules of the grad-norm table and the monotonicity check in
thesis/experiments/2026-10-14-stability-seed1-124m.md, reused as the failure
definition of thesis/experiments/2026-10-15-lr-headroom-124m.md.

    python scripts/stability_flags.py NAME=LOGS [NAME=LOGS ...] [--from 2000]

LOGS is one log or a comma-separated list in chunk order; on steps re-run
after a resume the later chunk wins. Prints one row per run: final val ppl
(the last validation), steps logged, count / first step / max of gradient
norms > 5, monotonicity violations, CLEAN or not, and the probe's M (max
|score| over layers) at step 2,000 and at the last probe when present.
"""
import argparse
import re

STEP = re.compile(r"^step +(\d+)/\d+ loss ([\d.]+) gnorm ([\d.]+)", re.M)
VAL = re.compile(r"^\s*\[val\] step (\d+) loss ([\d.]+) ppl ([\d.]+)", re.M)
PROBE = re.compile(r"^probe step (\d+) M ([\d.]+)", re.M)


def parse(paths):
    gn, val, probe = {}, {}, {}
    for path in paths.split(","):
        t = open(path, errors="ignore").read()
        gn.update({int(s): float(g) for s, _, g in STEP.findall(t)})
        val.update({int(s): float(p) for s, _, p in VAL.findall(t)})
        probe.update({int(s): float(m) for s, m in PROBE.findall(t)})
    return gn, val, probe


def flags(gn, val, probe, start):
    after = {s: g for s, g in gn.items() if s > start}
    big = sorted(s for s, g in after.items() if g > 5)
    vs = sorted(val)
    mono = [b for a, b in zip(vs, vs[1:]) if b > start and val[b] > 1.05 * val[a]]
    last = vs[-1] if vs else None
    return {
        "final_step": last,
        "final_ppl": val[last] if vs else float("nan"),
        "logged": len(after),
        "n_gn5": len(big),
        "first_gn5": big[0] if big else None,
        "max_gn": max(after.values()) if after else float("nan"),
        "n_mono": len(mono),
        "first_mono": mono[0] if mono else None,
        "clean": not big and not mono,
        "M_start": probe.get(start),
        "M_last": probe[max(probe)] if probe else None,
        "probe_last": max(probe) if probe else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=LOG[,LOG...]")
    ap.add_argument("--from", dest="start", type=int, default=2000)
    a = ap.parse_args()
    print(f"{'run':<22} {'final':>12} {'logged':>6} {'gn>5':>5} {'first':>6} "
          f"{'max gn':>8} {'mono':>4} {'first':>6}  verdict  probe M")
    for r in a.runs:
        name, paths = r.split("=", 1)
        f = flags(*parse(paths), a.start)
        fin = f"{f['final_ppl']:.2f}@{f['final_step']}"
        pm = ""
        if f["M_last"] is not None:
            m0 = "-" if f["M_start"] is None else f"{f['M_start']:.1f}"
            pm = f"{m0} @{a.start} -> {f['M_last']:.1f} @{f['probe_last']}"
        print(f"{name:<22} {fin:>12} {f['logged']:>6} {f['n_gn5']:>5} "
              f"{str(f['first_gn5'] or '-'):>6} {f['max_gn']:>8.2f} "
              f"{f['n_mono']:>4} {str(f['first_mono'] or '-'):>6}  "
              f"{'CLEAN' if f['clean'] else 'FAILED':<7}  {pm}")


if __name__ == "__main__":
    main()
