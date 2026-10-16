#!/usr/bin/env python3
"""--replay-rng check on the real 70/15/15 shards: advancing the sampler with
Shards.skip() must leave it exactly where Shards.batch() would, so a resumed
run draws the same next batches as an uninterrupted one. CPU only.
Run: python test_replay_rng.py"""
import importlib.util
import sys

import numpy as np
import torch

HERE = "/users/PAS2402/alexg/softmax/.worktrees/qknorm/triton/dev_scripts"
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("trainer", f"{HERE}/train_gpt2_stieltjes.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

MIX = ("web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7,"
       "math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15,"
       "code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15")
d = tr.Shards("train", MIX)
seed, bs, ctx, calls = 1 * 100003, 16, 1024, 96          # 96 calls = 3 steps x 32 accum
a = np.random.default_rng(seed)
for _ in range(calls):
    d.batch(bs, ctx, a, "cpu")                            # the training path
xa, ya = d.batch(bs, ctx, a, "cpu")
b = np.random.default_rng(seed)
for _ in range(calls):
    d.skip(bs, ctx, b)                                    # the replay path
xb, yb = d.batch(bs, ctx, b, "cpu")
assert torch.equal(xa, xb) and torch.equal(ya, yb), "skip() diverged from batch()"
assert a.bit_generator.state == b.bit_generator.state
c = np.random.default_rng(seed + 3)                       # the default reseed path differs
xc, _ = d.batch(bs, ctx, c, "cpu")
assert not torch.equal(xa, xc)
print(f"skip() == batch() after {calls} calls: next batch identical "
      f"({xa.shape[0]}x{xa.shape[1]} tokens), sampler states equal; "
      f"reseeding by step gives a different batch, as expected")
print("ALL OK")
