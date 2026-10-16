#!/usr/bin/env python3
"""Keep a verified, step-named copy of every checkpoint a job writes.

The trainers save in place, so each save overwrites the last; descriptive
analyses (scripts/probe_locate.py) need the intermediate ones. While JOB is
pending or running, poll CKPT: when it has changed and been quiet for
--settle seconds (no settle wait once the job has ended), read its step
(memory-mapped load), copy it to <CKPT without .pt>.step<N>.pt unless that
exists, verify the copy loads, and atomically put it in place. Exits after
a last pass once the job has ended. Login node.

    python scripts/snapshot_ckpts.py CKPT JOB [--every 60] [--settle 60]
"""
import argparse
import os
import shutil
import subprocess
import time

import torch

TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def step_of(path):
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return int(ck["step"])


def ended(job):
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-o", "State", "-j", job],
                         capture_output=True, text=True).stdout.strip()
    return bool(out) and out.split()[0].startswith(TERMINAL)


def snap(ckpt):
    try:
        step = step_of(ckpt)
    except Exception as e:                 # mid-write: try again next poll
        print(f"{time.strftime('%H:%M')} unreadable ({type(e).__name__}); retry", flush=True)
        return False
    dst = ckpt[:-3] + f".step{step}.pt"
    if os.path.exists(dst):
        return True
    tmp = dst + ".tmp"
    shutil.copyfile(ckpt, tmp)
    try:
        assert step_of(tmp) == step
    except Exception as e:
        os.remove(tmp)
        print(f"{time.strftime('%H:%M')} copy of step {step} failed to verify "
              f"({type(e).__name__}); retry", flush=True)
        return False
    os.replace(tmp, dst)
    print(f"{time.strftime('%H:%M')} saved {os.path.basename(dst)}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("job")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--settle", type=int, default=60)
    a = ap.parse_args()
    seen = None
    while True:
        done = ended(a.job)
        if os.path.exists(a.ckpt):
            mt = os.path.getmtime(a.ckpt)
            if mt != seen and (done or time.time() - mt >= a.settle):
                if snap(a.ckpt):
                    seen = mt
        if done:
            print(f"{time.strftime('%H:%M')} job {a.job} ended", flush=True)
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
