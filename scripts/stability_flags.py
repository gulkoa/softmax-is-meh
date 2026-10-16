#!/usr/bin/env python3
"""Stability flags for 124M pilot runs, from the training logs.

A run is CLEAN when it has a validation at --final (default 9,500) and,
after step --from (default 2,000, the end of warmup):
  - no logged gradient norm exceeds 5 (logged every 20 steps, clip 1.0),
  - no validation perplexity exceeds 1.05 x the previous validation, and
  - no logged loss, gradient norm or validation is nan/inf.
It FAILED on any of those events, and is INCOMPLETE (scored as failed once
the run has ended) when it has no events but never reached --final.
These are the rules of the grad-norm table and the monotonicity check in
thesis/experiments/2026-10-14-stability-seed1-124m.md, reused as the failure
definition of thesis/experiments/2026-10-15-lr-headroom-124m.md.

    python scripts/stability_flags.py NAME=LOGS [NAME=LOGS ...] [--from 2000]

LOGS is one log or a comma-separated list in chunk order; on steps re-run
after a resume the later chunk wins. Prints one row per run: final val ppl
(the last validation), steps logged, count / first step / max of gradient
norms > 5, monotonicity violations, the first nan/inf step, the verdict,
and the probe's M (max |score| over layers) at step 2,000 and at the last
probe when present, with the first probe step at which M exceeds 1,000
(the descriptive collapse-timing onset pinned in
thesis/experiments/2026-10-16-qk-collapse-survey.md). Warns when 20-step log lines are missing, and lists
every event (gradient norm > 5, nan/inf, validation jump) that an earlier
chunk logged but a later chunk's re-run of the same steps overwrote: shown,
not counted (the verdict follows the merged log).
"""
import argparse
import math
import re

# loss / gnorm / ppl may print as nan or inf: match any token, parse with float()
STEP = re.compile(r"^step +(\d+)/\d+ loss (\S+) gnorm (\S+)", re.M)
VAL = re.compile(r"^\s*\[val\] step (\d+) loss (\S+) ppl (\S+)", re.M)
PROBE = re.compile(r"^probe step (\d+) M ([\d.]+)", re.M)


def parse(paths):
    gn, loss, val, probe = {}, {}, {}, {}
    for path in paths.split(","):
        t = open(path, errors="ignore").read()
        for s, l, g in STEP.findall(t):
            gn[int(s)], loss[int(s)] = float(g), float(l)
        val.update({int(s): float(p) for s, _, p in VAL.findall(t)})
        probe.update({int(s): float(m) for s, m in PROBE.findall(t)})
    return gn, loss, val, probe


def events(gn, loss, val, start):
    """Set of (step, kind) failure events after `start` in one merged log."""
    ev = {(s, "gnorm>5") for s, g in gn.items() if s > start and g > 5}
    ev |= {(s, "nan/inf") for s in gn if s > start
           and not (math.isfinite(gn[s]) and math.isfinite(loss[s]))}
    ev |= {(s, "nan/inf val") for s in val if s > start and not math.isfinite(val[s])}
    vs = sorted(val)
    ev |= {(b, "val jump") for a, b in zip(vs, vs[1:])
           if b > start and val[b] > 1.05 * val[a]}
    return ev


def overwritten(paths, start):
    """Events present while an earlier chunk was the latest, gone from the
    final merge because a later chunk re-ran those steps."""
    chunks = paths.split(",")
    final = events(*parse(paths)[:3], start)
    lost = set()
    for i in range(1, len(chunks)):
        lost |= events(*parse(",".join(chunks[:i]))[:3], start) - final
    return sorted(lost)


def flags(gn, loss, val, probe, start, final):
    after = {s: g for s, g in gn.items() if s > start}
    big = sorted(s for s, g in after.items() if g > 5)
    vs = sorted(val)
    mono = [b for a, b in zip(vs, vs[1:]) if b > start and val[b] > 1.05 * val[a]]
    bad = sorted([s for s in gn if s > start
                  and not (math.isfinite(gn[s]) and math.isfinite(loss[s]))]
                 + [s for s in val if s > start and not math.isfinite(val[s])])
    last = vs[-1] if vs else None
    top = max(gn) if gn else start
    missing = [s for s in range(start + 20, top + 1, 20) if s not in gn]
    events = big or mono or bad
    verdict = ("FAILED" if events else
               "CLEAN" if final in val else "INCOMPLETE")
    return {
        "final_step": last,
        "final_ppl": val[last] if vs else float("nan"),
        "logged": len(after),
        "n_gn5": len(big),
        "first_gn5": big[0] if big else None,
        "max_gn": max((g for g in after.values() if math.isfinite(g)),
                      default=float("nan")),
        "n_mono": len(mono),
        "first_mono": mono[0] if mono else None,
        "first_bad": bad[0] if bad else None,
        "missing": len(missing),
        "verdict": verdict,
        "M_start": probe.get(start),
        "M_last": probe[max(probe)] if probe else None,
        "probe_last": max(probe) if probe else None,
        "M_onset": min((s for s, m in probe.items() if m > 1000), default=None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=LOG[,LOG...]")
    ap.add_argument("--from", dest="start", type=int, default=2000)
    ap.add_argument("--final", type=int, default=9500,
                    help="validation step a complete run must reach")
    a = ap.parse_args()
    print(f"{'run':<22} {'final':>12} {'logged':>6} {'gn>5':>5} {'first':>6} "
          f"{'max gn':>8} {'mono':>4} {'first':>6} {'nan':>6}  verdict     probe M")
    for r in a.runs:
        name, paths = r.split("=", 1)
        f = flags(*parse(paths), a.start, a.final)
        fin = f"{f['final_ppl']:.2f}@{f['final_step']}"
        pm = ""
        if f["M_last"] is not None:
            m0 = "-" if f["M_start"] is None else f"{f['M_start']:.1f}"
            pm = f"{m0} @{a.start} -> {f['M_last']:.1f} @{f['probe_last']}"
            pm += (f", M>1000 first @{f['M_onset']}" if f["M_onset"] is not None
                   else ", M never >1000")
        print(f"{name:<22} {fin:>12} {f['logged']:>6} {f['n_gn5']:>5} "
              f"{str(f['first_gn5'] or '-'):>6} {f['max_gn']:>8.2f} "
              f"{f['n_mono']:>4} {str(f['first_mono'] or '-'):>6} "
              f"{str(f['first_bad'] or '-'):>6}  {f['verdict']:<10}  {pm}")
        if f["missing"]:
            print(f"  WARNING {name}: {f['missing']} of the 20-step log lines "
                  f"after step {a.start} are missing")
        if f["logged"] == 0:
            print(f"  WARNING {name}: no gradient-norm lines after step "
                  f"{a.start}; the gradient-norm rule was not applied")
        lost = overwritten(paths, a.start)
        if lost:
            print(f"  NOTE {name}: {len(lost)} event(s) overwritten by a later "
                  "chunk's re-run (shown, not counted): "
                  + ", ".join(f"{k} @{s}" for s, k in lost))


if __name__ == "__main__":
    main()
