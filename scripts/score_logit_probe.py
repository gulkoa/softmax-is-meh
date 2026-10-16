#!/usr/bin/env python3
"""Score the pre-registered logit-growth probe
(thesis/experiments/2026-10-15-logit-growth-probe.md) from the two training logs.

M(t)  = max over layers of max |score| on the fixed probe sequence at step t.
t*    = the softmax run's first logged gradient norm > 5 after step 2,000.
G_soft = M_soft(t* - 20) / M_soft(2000)
G_stj  = max_{2000 <= t <= t* - 20} M_stj(t) / M_stj(2000)
Readings: 1 G_soft >= 2 and G_stj <= 1.5 -> Stieltjes does not grow;
          2 G_soft >= 2 and G_stj >= 2   -> Stieltjes tolerates growth;
          3 G_soft < 2                   -> logit growth is not the trigger;
          4 no softmax spike before the stop step -> no reading;
          5 otherwise                    -> partial.

    python scripts/score_logit_probe.py SOFTMAX_LOGS STIELTJES_LOGS [--from 2000]

Each LOGS argument is one log or a comma-separated list in chunk order; on
steps re-run after a resume, the later chunk wins (it carries the trajectory).
"""
import argparse
import re

PROBE = re.compile(r"^probe step (\d+) M ([\d.]+) maxabs ([\d. ]+)\| minent ([\d. ]+)$", re.M)
STEP = re.compile(r"^step +(\d+)/\d+ loss ([\d.]+) gnorm ([\d.]+)", re.M)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("softmax_log")
    ap.add_argument("stj_log")
    ap.add_argument("--from", type=int, default=2000, dest="t0")
    a = ap.parse_args()
    P_s, G_s, stop_s = parse(a.softmax_log)
    P_j, G_j, stop_j = parse(a.stj_log)
    print(f"softmax: {len(P_s)} probes (last {max(P_s, default=None)}), stopped={stop_s}; "
          f"Stieltjes: {len(P_j)} probes (last {max(P_j, default=None)}), stopped={stop_j}")
    t0 = a.t0
    if t0 not in P_s or t0 not in P_j:
        print(f"no probe at step {t0} in both runs yet: nothing to score")
        return
    tstar = next((s for s in sorted(G_s) if s > t0 and G_s[s] > 5), None)
    jstar = next((s for s in sorted(G_j) if s > t0 and G_j[s] > 5), None)
    print(f"softmax first logged grad norm > 5 after {t0}: {tstar}; Stieltjes: {jstar}")
    if tstar is None:
        print("READING 4: no softmax spike before the stop step -> no reading")
        return
    pre = tstar - 20
    G_soft = P_s[pre][0] / P_s[t0][0]
    win = [P_j[s][0] for s in sorted(P_j) if t0 <= s <= pre]
    G_stj = max(win) / P_j[t0][0]
    print(f"M_soft({t0}) {P_s[t0][0]:.3f} -> M_soft({pre}) {P_s[pre][0]:.3f}: G_soft = {G_soft:.3f}")
    print(f"M_stj({t0}) {P_j[t0][0]:.3f}, max over [{t0}, {pre}] {max(win):.3f}: G_stj = {G_stj:.3f}")
    if G_soft < 2:
        r = "3: logit growth is not the trigger (G_soft < 2)"
    elif G_stj <= 1.5:
        r = "1: Stieltjes does not grow (G_soft >= 2, G_stj <= 1.5)"
    elif G_stj >= 2:
        r = "2: Stieltjes tolerates growth (both >= 2)"
    else:
        r = "5: partial (G_soft >= 2, 1.5 < G_stj < 2)"
    print("PRE-REGISTERED READING", r)
    # descriptive: per-layer growth and entropy at the same steps
    print("descriptive, per layer: max|score| at t0 -> t*-20 | min-head entropy at t0 -> t*-20")
    for name, P in (("softmax", P_s), ("Stieltjes", P_j)):
        a0, a1 = P[t0][1], P[pre][1]
        e0, e1 = P[t0][2], P[pre][2]
        print(f"  {name:9} max|s| " + " ".join(f"{x:.1f}>{y:.1f}" for x, y in zip(a0, a1)))
        print(f"  {name:9} ent    " + " ".join(f"{x:.2f}>{y:.2f}" for x, y in zip(e0, e1)))


if __name__ == "__main__":
    main()
