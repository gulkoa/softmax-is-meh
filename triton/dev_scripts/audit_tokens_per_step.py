"""Token-accounting audit (advisor request, 2026-08-25 meeting): pull ONE
batch from the real training data loader and count tokens — no model,
no GPU. Confirms tokens/optimizer-step = micro_bs * grad_accum * ctx.

Usage: python audit_tokens_per_step.py [--data-mix SPEC] [--micro-bs 8]
                                       [--grad-accum 64] [--ctx 1024]
                                       [--total-tokens 15e9]
SPEC as in train_gpt2_stieltjes.py ('dir' or 'name=dir:w,name2=dir2:w').
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import Shards  # noqa: E402  (the real loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-mix", default=None, dest="data_mix")
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--total-tokens", type=float, default=15e9)
    a = ap.parse_args()

    d = Shards("train", a.data_mix)
    rng = np.random.default_rng(0)
    x, y = d.batch(a.micro_bs, a.ctx, rng, torch.device("cpu"))
    per_micro = x.numel()
    per_step = per_micro * a.grad_accum
    print(f"one micro-batch: x{tuple(x.shape)} y{tuple(y.shape)} "
          f"-> {per_micro} tokens")
    print(f"x grad_accum {a.grad_accum} -> {per_step} tokens / optimizer step")
    print(f"{a.total_tokens:.3g} tokens // {per_step} -> "
          f"{int(a.total_tokens // per_step)} optimizer steps")
    print("targets = inputs shifted by one:",
          bool((x[:, 1:] == y[:, :-1]).all()))
    print(f"token ids: dtype {x.dtype}, max {int(x.max())} (GPT-2 vocab 50257)")
    print("NOTE: the trainer logs to wandb once per 20 steps (+1 per val), "
          "WITHOUT step=; wandb's default x-axis '_step' is therefore the "
          "log-call index (~20 trainer steps ≈ 10.5M tokens per unit). "
          "Plot against the logged 'step' field.")


if __name__ == "__main__":
    main()
