#!/usr/bin/env python3
"""Keep one verified backup of each training checkpoint (<ckpt>.bak).

The trainers save in place (no atomic rename, no SIGTERM handling), so a
job killed by its time limit mid-save would leave a truncated checkpoint
and no other copy. This copies a checkpoint only when it is newer than its
backup and has been quiet for --settle seconds, verifies the COPY loads
(memory-mapped, reads "step"), then atomically swaps it in. A bad copy never
replaces a good backup. Login node; run it between or during chunks.

    python scripts/backup_ckpts.py CKPT [CKPT ...] [--settle 120]
"""
import argparse
import os
import shutil
import sys
import time

import torch


def verify(path):
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return int(ck["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--settle", type=int, default=120)
    a = ap.parse_args()
    rc = 0
    for c in a.ckpts:
        bak = c + ".bak"
        name = os.path.basename(c)
        if not os.path.exists(c):
            print(f"{name}: missing")
            rc = 1
            continue
        age = time.time() - os.path.getmtime(c)
        if os.path.exists(bak) and os.path.getmtime(bak) >= os.path.getmtime(c):
            print(f"{name}: backup current (step {verify(bak)})")
            continue
        if age < a.settle:
            print(f"{name}: written {age:.0f}s ago, not settled; skip")
            continue
        tmp = bak + ".tmp"
        shutil.copyfile(c, tmp)
        try:
            step = verify(tmp)
        except Exception as e:  # truncated or mid-write copy
            os.remove(tmp)
            print(f"{name}: copy FAILED to verify ({type(e).__name__}); backup kept")
            rc = 1
            continue
        os.replace(tmp, bak)
        print(f"{name}: backed up step {step}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
