"""
Eval-only recovery for MQMTAR v7 checkpoints: load a saved model and run
exact-match generation eval on selected length splits (used to recover the
long dense splits that 4h probe walltimes cut off; v7 saves the checkpoint
BEFORE the eval loop).

Usage:
  python mqmtar_eval_ckpt.py <ckpt.pt> [<ckpt2.pt> ...] \
      --lengths 4096 8192 [--data DIR]
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mqmtar_headtohead_v7 as M  # noqa: E402  (frozen file: import only)

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

SPLIT_INDEX = {64: 0, 128: 1, 256: 2, 512: 3, 1024: 4, 2048: 5, 4096: 6,
               8192: 7, 16384: 8, 32768: 9, 65536: 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--data", default=os.environ.get(
        "MQMTAR_DATA",
        "/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/"
        "50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4"))
    args = ap.parse_args()
    device = torch.device("cuda")

    splits = {}
    for L in args.lengths:
        prefix = os.path.join(args.data, f"test_{SPLIT_INDEX[L]}_{L}")
        splits[L] = M.read_split(prefix)
        print(f"loaded split {L}", flush=True)

    for path in args.ckpts:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SimpleNamespace(**blob["args"])
        label = getattr(cfg, "arm_label", cfg.arm)
        model = M.MQMTARModel(cfg).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        run = wandb.init(
            project="stieltjes-flash-attn",
            name=f"mqmtar-evalckpt-{label}-"
                 f"{os.environ.get('SLURM_JOB_ID', 'local')}",
            config={**blob["args"], "recovered_from": path},
            reinit=True,
        )
        print(f"\n===== {label}  ({os.path.basename(path)}) =====", flush=True)
        for L in args.lengths:
            s, t = splits[L]
            n = 1000 if L <= 2048 else (500 if L == 4096 else 100)
            bs = max(1, min(64, 2 ** 27 // (cfg.n_head * L * L)))
            t0 = time.time()
            acc = M.eval_split(model, s, t, device, n, bs)
            print(f"  len {L:6d}: acc {acc:.4f}  (n={n}, "
                  f"{time.time()-t0:.0f}s)", flush=True)
            run.log({f"final/acc_{L}": acc})
            run.summary[f"acc_{L}"] = acc
        run.finish()
        del model
        torch.cuda.empty_cache()

    print("DONE")


if __name__ == "__main__":
    main()
