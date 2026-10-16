#!/usr/bin/env python3
"""Score the pre-registered LR-headroom sweep
(thesis/experiments/2026-10-15-lr-headroom-124m.md) from the training logs.

Reads the job list ("NAME JOB [JOB ...]" per run, chunk order; the
conditional stage appends A@8e-3 / D@8e-3 rows), merges each run's chunk
logs (later chunk wins on re-run steps), flags it with
stability_flags.flags, and applies readings Q1-Q4, the attribution rule
and the conditional-stage trigger exactly as pre-registered. A run whose
jobs are still queued or running and that has no failure event yet is
PENDING, and every reading that needs it is withheld; one failure event
is final (the rule needs only one), and a run that ended without reaching
step 9,500 counts as failed.

    python scripts/score_lr_headroom.py [--jobs results/lr_headroom_jobs.txt]
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stability_flags import flags, overwritten, parse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")
# existing grid, seed 0 (scripts/stability_flags.py on the logs in the record)
A_DONE = {6e-4: ("clean", 16.89), 1e-3: ("clean", 16.30)}
A_6E4 = 16.89
LR = {"2e-3": 2e-3, "4e-3": 4e-3, "8e-3": 8e-3, "3e-4": 3e-4, "6e-4": 6e-4}


def ended(jobs):
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-o", "State", "-j",
                          ",".join(jobs)], capture_output=True, text=True).stdout
    states = [ln.split()[0] for ln in out.splitlines() if ln.strip()]
    return len(states) == len(jobs) and all(s.startswith(TERMINAL) for s in states)


def status(f, done):
    """clean / failed / pending from the stability flags and whether the run ended."""
    if f["verdict"] == "FAILED":
        return "failed"
    if f["verdict"] == "CLEAN":
        return "clean"
    return "failed" if done else "pending"      # INCOMPLETE


def first_event(f):
    ev = [s for s in (f["first_gn5"], f["first_mono"], f["first_bad"]) if s is not None]
    return min(ev) if ev else None


def attribute(f, probe):
    """Pre-registered: attention-logit-driven iff M(last probe before the first
    failure event) >= 2 x M(1,000 steps earlier)."""
    e = first_event(f)
    if e is None:
        return "no failure event (crash or unfinished): not attributable"
    before = [s for s in probe if s < e]
    if not before:
        return f"no probe before the first event at step {e}"
    p = max(before)
    if p - 1000 not in probe:
        return f"no probe at step {p - 1000}"
    r = probe[p] / probe[p - 1000]
    verdict = ("attention-logit-driven" if r >= 2 else
               "not attributed to attention logits")
    return (f"first event step {e}; M({p}) {probe[p]:.1f} / M({p - 1000}) "
            f"{probe[p - 1000]:.1f} = {r:.2f} -> {verdict}")


def readings(R):
    """R: name -> (status, final_ppl or None, attribution text or None)."""
    out = []
    st = lambda n: R[n][0] if n in R else "not run"      # noqa: E731
    ppl = lambda n: R[n][1]                              # noqa: E731

    a2, a4, a8 = st("A@2e-3"), st("A@4e-3"), st("A@8e-3")
    if "pending" in (a2,) or (a2 == "clean" and a4 == "pending"):
        out.append("Q1 pending")
    elif a2 == "failed":
        out.append("Q1 H3 narrow: A failed at 2e-3 (headroom between 1x and 2x)")
    elif a4 == "failed":
        out.append("Q1 H2 moderate: A clean at 2e-3, failed at 4e-3")
    else:
        tail = {"clean": "; also clean at 8e-3: headroom >= 8x (lower bound)",
                "failed": "; failed at 8e-3", "pending": "; 8e-3 pending",
                "not run": ""}[a8]
        out.append("Q1 H1 wide: A clean at 2e-3 and 4e-3 (>= 4x)" + tail)

    a_clean = [lr for lr, (s, _) in A_DONE.items() if s == "clean"]
    a_clean += [LR[n.split("@")[1]] for n in ("A@2e-3", "A@4e-3", "A@8e-3")
                if st(n) == "clean"]
    a_pending = any(st(n) == "pending" for n in ("A@2e-3", "A@4e-3", "A@8e-3"))
    a_never_failed = not any(st(n) == "failed" for n in ("A@2e-3", "A@4e-3", "A@8e-3"))
    l3, l6 = st("LS@3e-4"), st("LS@6e-4")
    if "pending" in (l3, l6) or a_pending:
        out.append("Q2 pending")
    else:
        top = 6e-4 if l6 == "clean" else 3e-4 if l3 == "clean" else None
        if top is None:
            out.append("Q2 LS failed at 3e-4 and 6e-4 (and 1e-3): ratio undefined")
        else:
            r = max(a_clean) / top
            bound = " (lower bound: A never failed)" if a_never_failed else ""
            out.append(f"Q2 highest clean LS lr {top:g}; Stieltjes / matched "
                       f"softmax = {max(a_clean):g} / {top:g} = {r:.2f}x{bound}")

    for lr in ("2e-3", "4e-3", "8e-3"):
        a, d = st(f"A@{lr}"), st(f"D@{lr}")
        if a == "not run":
            continue
        if d == "not run":
            out.append(f"Q3 @{lr}: D not run (failed at the previous lr)")
            continue
        if "pending" in (a, d):
            out.append(f"Q3 @{lr} pending")
            continue
        if a == "clean" and d == "clean":
            r = ppl(f"A@{lr}") / ppl(f"D@{lr}")
            v = ("matches the cure" if r <= 1.03 else
                 "the norm is more robust" if r > 1.05 else "partial")
            out.append(f"Q3 @{lr} {v}: both clean, A/D = {r:.3f}")
        elif d == "clean":
            out.append(f"Q3 @{lr} the norm is more robust: D clean, A failed")
        elif a == "clean":
            out.append(f"Q3 @{lr} Stieltjes is more robust: A clean, D failed")
        else:
            out.append(f"Q3 @{lr} partial: both failed")

    if l6 == "pending":
        out.append("Q4 pending")
    elif ppl("LS@6e-4") is None:
        out.append("Q4 LS@6e-4 has no step-9,500 score (failed before it): "
                   "outside the pre-registered rule, reported as such")
    else:
        r = ppl("LS@6e-4") / A_6E4
        v = ("caveat removed at 6e-4" if r >= 1.15 else
             "the learnable scale stabilizes softmax at 6e-4" if r < 1.05
             else "partial")
        out.append(f"Q4 {v}: LS@6e-4 {ppl('LS@6e-4'):.2f} / A@6e-4 "
                   f"{A_6E4} = {r:.3f} (LS@6e-4 {l6})")

    for n in ("A@2e-3", "A@4e-3", "A@8e-3"):
        if st(n) == "failed":
            out.append(f"attribution {n}: {R[n][2]}")

    if a4 == "clean" and a8 == "not run":
        out.append("CONDITIONAL STAGE TRIGGERED: run A@8e-3"
                   + (" and D@8e-3" if st("D@4e-3") == "clean" else
                      " (not D@8e-3: D not clean at 4e-3)" if st("D@4e-3") == "failed"
                      else " (D@8e-3 waits on D@4e-3)"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=os.path.join(ROOT, "results", "lr_headroom_jobs.txt"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--assume-ended", action="store_true",
                    help="treat every run as ended (tests; no sacct)")
    a = ap.parse_args()
    R = {}
    print(f"{'run':<9} {'status':<8} {'final':>12} {'gn>5':>5} {'first':>6} "
          f"{'max gn':>8} {'mono':>4} {'nan':>6}  probe M")
    for line in open(a.jobs):
        if not line.strip():
            continue
        name, *jobs = line.split()
        logs = [os.path.join(a.results, f"qknorm-pilot-{j}.out") for j in jobs]
        logs = [p for p in logs if os.path.exists(p)]
        done = a.assume_ended or ended(jobs)
        if not logs:
            R[name] = ("failed" if done else "pending", None, None)
            print(f"{name:<9} {R[name][0]:<8} (no log yet)")
            continue
        gn, loss, val, probe = parse(",".join(logs))
        f = flags(gn, loss, val, probe, 2000, 9500)
        s = status(f, done)
        final = f["final_ppl"] if f["final_step"] == 9500 else None
        R[name] = (s, final, attribute(f, probe) if s == "failed" else None)
        pm = (f"{f['M_start'] if f['M_start'] is not None else '-'} @2000 -> "
              f"{f['M_last']:.1f} @{f['probe_last']}" if f["M_last"] is not None else "")
        print(f"{name:<9} {s:<8} {f['final_ppl']:>7.2f}@{str(f['final_step']):<5}"
              f"{f['n_gn5']:>5} {str(f['first_gn5'] or '-'):>6} {f['max_gn']:>8.2f} "
              f"{f['n_mono']:>4} {str(f['first_bad'] or '-'):>6}  {pm}")
        if f["first_bad"] is not None and not done:
            print(f"  EARLY CANCEL {name}: nan/inf at step {f['first_bad']}; "
                  f"scancel the remaining chunks of {' '.join(jobs)}")
        if f["missing"]:
            print(f"  WARNING {name}: {f['missing']} 20-step log lines missing")
        lost = overwritten(",".join(logs), 2000)
        if lost:
            print(f"  NOTE {name}: {len(lost)} event(s) overwritten by a later "
                  "chunk's re-run (shown, not counted): "
                  + ", ".join(f"{k} @{s}" for s, k in lost))
    print()
    for r in readings(R):
        print(r)


if __name__ == "__main__":
    main()
