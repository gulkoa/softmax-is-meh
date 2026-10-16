#!/usr/bin/env python3
"""Score the pre-registered logit-growth probe
(thesis/experiments/2026-10-15-logit-growth-probe.md) from the two training logs.

M(t)  = max over layers of max |score| on the fixed probe sequence at step t.
t*    = the softmax run's first logged gradient norm > 5 after step 2,000
        (a nan/inf gradient norm counts as > 5).
G_soft = M_soft(t* - 20) / M_soft(2000)
G_stj  = max_{2000 <= t <= t* - 20} M_stj(t) / M_stj(2000)
Readings: 1 G_soft >= 2 and G_stj <= 1.5 -> Stieltjes does not grow;
          2 G_soft >= 2 and G_stj >= 2   -> Stieltjes tolerates growth;
          3 G_soft < 2                   -> logit growth is not the trigger;
          4 no softmax spike before the stop step -> no reading;
          5 otherwise                    -> partial.
Reading 4 needs the softmax run to have REACHED its stop step ("STOPPED at
step" / "already stopped" in its logs); a run that ended short of both a
spike and the stop step is INCOMPLETE (append a --replay-rng chunk). If the
probe at t* - 20 is missing, the last probe before t* is used (and said).
G is scored from the logged values; a G within 1% of a bar it is read
against is flagged "at the bar" (the reading itself is unchanged).

    python scripts/score_logit_probe.py SOFTMAX_LOGS STIELTJES_LOGS [--from 2000]

Each LOGS argument is one log or a comma-separated list in chunk order; on
steps re-run after a resume, the later chunk wins (it carries the trajectory).
nan/inf values in the logs are parsed, never dropped.
"""
import argparse
import math
import re

PROBE = re.compile(r"^probe step (\d+) M (\S+) maxabs (.*?) \| minent (.*)$", re.M)
STEP = re.compile(r"^step +(\d+)/\d+ loss (\S+) gnorm (\S+)", re.M)


def parse(paths):
    probes, gn, stopped = {}, {}, False
    for path in paths.split(","):
        t = open(path, errors="ignore").read()
        probes.update({int(s): (float(m), [float(v) for v in a.split()],
                                [float(v) for v in e.split()])
                       for s, m, a, e in PROBE.findall(t)})
        gn.update({int(s): float(g) for s, _, g in STEP.findall(t)})
        stopped = stopped or "STOPPED at step" in t or "already stopped" in t
    return probes, gn, stopped


def spike(g):
    return not math.isfinite(g) or g > 5


def near(g, bar):
    return abs(g - bar) <= 0.01 * bar


def score(softmax_logs, stj_logs, t0=2000):
    """Returns (reading, printed lines); reading is 1-5, "INCOMPLETE" or None."""
    P_s, G_s, stop_s = parse(softmax_logs)
    P_j, G_j, stop_j = parse(stj_logs)
    out = [f"softmax: {len(P_s)} probes (last {max(P_s, default=None)}), stopped={stop_s}; "
           f"Stieltjes: {len(P_j)} probes (last {max(P_j, default=None)}), stopped={stop_j}"]
    if t0 not in P_s or t0 not in P_j:
        return None, out + [f"no probe at step {t0} in both runs yet: nothing to score"]
    tstar = next((s for s in sorted(G_s) if s > t0 and spike(G_s[s])), None)
    jstar = next((s for s in sorted(G_j) if s > t0 and spike(G_j[s])), None)
    out.append(f"softmax first logged grad norm > 5 after {t0}: {tstar}; Stieltjes: {jstar}")
    if tstar is None:
        if not stop_s:
            return "INCOMPLETE", out + [
                f"INCOMPLETE: softmax ended at step {max(G_s)} without a spike and "
                "short of its stop step -> append a --replay-rng chunk"]
        return 4, out + ["READING 4: no softmax spike before the stop step -> no reading"]
    pre = max(s for s in P_s if s < tstar)
    if pre != tstar - 20:
        out.append(f"probe at t*-20 = {tstar - 20} missing: using the last probe before t*, {pre}")
    G_soft = P_s[pre][0] / P_s[t0][0]
    win = [P_j[s][0] for s in sorted(P_j) if t0 <= s <= pre]
    if max(P_j) < pre:
        out.append(f"WARNING: Stieltjes probes end at {max(P_j)}, before t*-20 = {pre}")
    G_stj = max(win) / P_j[t0][0]
    out.append(f"M_soft({t0}) {P_s[t0][0]:.3f} -> M_soft({pre}) {P_s[pre][0]:.3f}: G_soft = {G_soft:.3f}")
    out.append(f"M_stj({t0}) {P_j[t0][0]:.3f}, max over [{t0}, {pre}] {max(win):.3f}: G_stj = {G_stj:.3f}")
    if G_soft < 2:
        r, txt, bars = 3, "logit growth is not the trigger (G_soft < 2)", [(G_soft, 2)]
    elif G_stj <= 1.5:
        r, txt, bars = 1, "Stieltjes does not grow (G_soft >= 2, G_stj <= 1.5)", [(G_soft, 2), (G_stj, 1.5)]
    elif G_stj >= 2:
        r, txt, bars = 2, "Stieltjes tolerates growth (both >= 2)", [(G_soft, 2), (G_stj, 2)]
    else:
        r, txt, bars = 5, "partial (G_soft >= 2, 1.5 < G_stj < 2)", [(G_soft, 2), (G_stj, 1.5), (G_stj, 2)]
    flag = [f"{g:.3f} vs {b}" for g, b in bars if near(g, b)]
    out.append(f"PRE-REGISTERED READING {r}: {txt}"
               + (f"  [at the bar: {', '.join(flag)}]" if flag else ""))
    # descriptive: per-layer growth and entropy at the same steps
    out.append(f"descriptive, per layer: max|score| at {t0} -> {pre} | min-head entropy at {t0} -> {pre}"
               " (entropy from bf16 scores: unreliable where |score| is in the thousands)")
    for name, P in (("softmax", P_s), ("Stieltjes", P_j)):
        p1 = pre if pre in P else max(s for s in P if s <= pre)
        a0, a1 = P[t0][1], P[p1][1]
        e0, e1 = P[t0][2], P[p1][2]
        out.append(f"  {name:9} max|s| " + " ".join(f"{x:.1f}>{y:.1f}" for x, y in zip(a0, a1)))
        out.append(f"  {name:9} ent    " + " ".join(f"{x:.2f}>{y:.2f}" for x, y in zip(e0, e1)))
    return r, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("softmax_log")
    ap.add_argument("stj_log")
    ap.add_argument("--from", type=int, default=2000, dest="t0")
    a = ap.parse_args()
    for line in score(a.softmax_log, a.stj_log, a.t0)[1]:
        print(line)


if __name__ == "__main__":
    main()
