"""Cold-read cost of one training step's windows: the synchronous path
(Shards.batch: serial memmap slices) vs the read-ahead loader's reads
(one step's windows in flight with os.pread on a thread pool).

Fresh seeds every round (cached windows would time the page cache), and
the two methods alternate which goes first. Light I/O (~1 MB per method
per round), fine on a login node. Numbers cited in the stilt.1.2 plan
(2026-10-16, login node, stilt.1.2 mix, 355M step = 64 x 8 windows):
serial 2.5-7.0 s/step, 32-thread pread 1.0-1.2 s/step.

    python scripts/bench_shard_reads.py [--mix SPEC] [--rounds 4]
        [--micro-bs 8] [--grad-accum 64] [--ctx 1024] [--threads 32]
"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

B = "/fs/scratch/PAS2836/alexg"
MIX = (f"web={B}/fineweb_edu_10bt_tok32k:0.7,math={B}/finemath_4plus_tok32k:0.15,"
       f"code={B}/codeparrot_py_tok32k:0.15")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=MIX)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    dirs, w = [], []
    for part in a.mix.split(","):
        d, wt = part.split("=", 1)[1].rsplit(":", 1)
        dirs.append(d); w.append(float(wt))
    w = np.asarray(w) / sum(w)
    paths = [sorted(glob.glob(f"{d}/shard_*.bin"))[:-1] for d in dirs]   # train shards
    maps = [[np.memmap(p, dtype=np.uint16, mode="r") for p in ps] for ps in paths]
    fds = [[os.open(p, os.O_RDONLY) for p in ps] for ps in paths]
    ctx, n = a.ctx, a.micro_bs * a.grad_accum
    pool = ThreadPoolExecutor(a.threads)

    def serial(rng):
        for _ in range(a.grad_accum):
            xs = np.empty((a.micro_bs, ctx + 1), dtype=np.int64)
            for i in range(a.micro_bs):
                s = rng.choice(len(maps), p=w); m = maps[s][rng.integers(len(maps[s]))]
                off = rng.integers(0, len(m) - ctx - 1); xs[i] = m[off:off + ctx + 1]

    def pooled(rng):
        rows = []
        for _ in range(n):
            s = int(rng.choice(len(maps), p=w)); k = int(rng.integers(len(maps[s])))
            rows.append((s, k, int(rng.integers(0, len(maps[s][k]) - ctx - 1))))
        raw = b"".join(pool.map(
            lambda r: os.pread(fds[r[0]][r[1]], 2 * (ctx + 1), 2 * r[2]), rows))
        np.frombuffer(raw, dtype=np.uint16).astype(np.int64)

    base = int(time.time() * 1000) % 10**9
    print(f"{n} windows of {ctx + 1} tokens per step, {len(dirs)} sources")
    for r in range(a.rounds):
        order = (serial, pooled) if r % 2 == 0 else (pooled, serial)
        res = {}
        for j, f in enumerate(order):
            t = time.time(); f(np.random.default_rng(base + 10 * r + j))
            res[f.__name__] = time.time() - t
        print(f"round {r}: serial memmap {res['serial']:.3f} s/step, "
              f"{a.threads}-thread pread {res['pooled']:.3f} s/step", flush=True)


if __name__ == "__main__":
    main()
