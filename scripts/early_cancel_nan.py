#!/usr/bin/env python3
"""Early-cancel check for chained runs, the pre-registered rule of
thesis/experiments/2026-10-15-lr-headroom-124m.md: a run whose merged
chunk logs show a nan/inf loss, gradient norm or validation is failed at
that step, and its chunks that are still queued or running are cancelled.
For runs kept out of results/lr_headroom_jobs.txt (that file belongs to
the seed-0 sweep and its scorer, score_lr_headroom.py).

JOBSFILE lines: NAME JOB1 ... JOBn (chunk order; logs are
results/qknorm-pilot-<job>.out). Prints one line per run; with --cancel,
scancels the queued or running jobs of every run that shows a nan/inf.
    python scripts/early_cancel_nan.py JOBSFILE [--results DIR] [--cancel]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stability_flags import flags, parse  # noqa: E402

ROOT = os.path.dirname(HERE)


def nan_runs(jobsfile, results):
    """[(name, first nan/inf step or None, jobs)] for every run in JOBSFILE."""
    out = []
    for line in open(jobsfile):
        if not line.strip():
            continue
        name, *jobs = line.split()
        logs = [os.path.join(results, f"qknorm-pilot-{j}.out") for j in jobs]
        logs = [p for p in logs if os.path.exists(p)]
        bad = flags(*parse(",".join(logs)), 2000, 9500)["first_bad"] if logs else None
        out.append((name, bad, jobs))
    return out


def live(jobs):
    """The jobs among these that are still queued or running."""
    r = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(jobs)],
                       capture_output=True, text=True)
    return [j for j in r.stdout.split() if j in jobs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobsfile")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cancel", action="store_true")
    a = ap.parse_args()
    for name, bad, jobs in nan_runs(a.jobsfile, a.results):
        if bad is None:
            print(f"{name}: no nan/inf")
            continue
        print(f"EARLY CANCEL {name}: nan/inf at step {bad}")
        if a.cancel:
            todo = live(jobs)
            if todo:
                subprocess.run(["scancel", *todo], check=False)
                print(f"  cancelled {' '.join(todo)}")


if __name__ == "__main__":
    main()
