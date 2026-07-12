"""
SCALE SWEEP: does making the model bigger change the softmax-vs-Stieltjes
OOD verdict on max-retrieval?

Motivation: the 2026-04-20 finding showed q=2's extrapolation collapse was
CAPACITY-GATED (1-head collapsed; 4-head matched softmax). The 2026-07-12
systematic comparison (d_emb=128) has softmax ahead by 4-10pp beyond 32x.
Here: d_emb in {256, 512} (vs the 128 baseline already measured), same
3 seeds, same protocol, eval to L=8192.

Arms per (d_emb, seed):
  softmax          (PyTorch, Jack's mapping)
  stj-q4 / stj-q2  (PyTorch normalized Stieltjes — algorithmic question;
                    kernel fidelity is already established)
  flashn-q4        only at d_emb=256 (kernel head_dim limit 256)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts")

from flashnorm_archnew import (  # noqa: E402
    build_ref, build_flash, train, evaluate, SimplexMappingEnum)

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

SEEDS = [0, 1, 2]
DEMBS = [int(x) for x in os.environ.get("SWEEP_DEMBS", "256,512").split(",")]
STEPS = 3000
ID_LEN = 16
LENGTHS = [16, 128, 256, 512, 1024, 2048, 4096, 8192]
N_CLASSES = 10
ITEM_DIM = 1 + N_CLASSES
NUM_ITER = 8


def run_one(model, seed, device):
    train(model, seq_len=ID_LEN, n_classes=N_CLASSES, device=device,
          steps=STEPS, bs=256, lr=1e-3, wd=1e-4,
          warmup=max(1, STEPS // 10), seed=seed)
    return {L: evaluate(model, seq_len=L, n_classes=N_CLASSES, device=device,
                        samples=2048 if L == ID_LEN else 1024, bs=256)
            for L in LENGTHS}


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  dembs={DEMBS}")
    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"maxretr-scale-sweep-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config=dict(seeds=SEEDS, dembs=DEMBS, steps=STEPS, id_len=ID_LEN,
                    lengths=LENGTHS, num_iter=NUM_ITER),
    )

    acc = {}

    def record(demb, label, accs, seed):
        key = (demb, label)
        acc.setdefault(key, {L: [] for L in LENGTHS})
        for L in LENGTHS:
            acc[key][L].append(accs[L])
        print(f"d={demb} seed={seed} {label:10s}: "
              + " ".join(f"L{L}={accs[L]:.1f}" for L in LENGTHS), flush=True)
        run.log({"d_emb": demb, "seed": seed, "label": label,
                 **{f"acc/d{demb}/{label}/L{L}": accs[L] for L in LENGTHS}})

    for demb in DEMBS:
        for seed in SEEDS:
            model = build_ref(SimplexMappingEnum.softmax, device, d_emb=demb,
                              n_classes=N_CLASSES, item_dim=ITEM_DIM, sq=4.0,
                              seed=seed)
            record(demb, "softmax", run_one(model, seed, device), seed)
            del model
            torch.cuda.empty_cache()

            for sq in [2.0, 4.0]:
                model = build_ref(SimplexMappingEnum.stieltjes, device,
                                  d_emb=demb, n_classes=N_CLASSES,
                                  item_dim=ITEM_DIM, sq=sq, seed=seed)
                record(demb, f"stj-q{sq:g}", run_one(model, seed, device), seed)
                del model
                torch.cuda.empty_cache()

            if demb <= 256:
                model = build_flash(device, d_emb=demb, n_classes=N_CLASSES,
                                    item_dim=ITEM_DIM, sq=4.0,
                                    num_iter=NUM_ITER, seed=seed)
                record(demb, "flashn-q4", run_one(model, seed, device), seed)
                del model
                torch.cuda.empty_cache()

    for demb in DEMBS:
        labels = [lab for (d, lab) in acc if d == demb]
        labels = sorted(set(labels), key=lambda s: (s != "softmax", s))
        print(f"\n===== d_emb={demb} mean±std over seeds {SEEDS} =====")
        table = wandb.Table(columns=["variant"] + [f"L{L}" for L in LENGTHS])
        for label in labels:
            cells = []
            for L in LENGTHS:
                v = np.array(acc[(demb, label)][L])
                cells.append(f"{v.mean():6.2f}±{v.std():4.2f}")
                run.summary[f"mean/d{demb}/{label}/L{L}"] = float(v.mean())
            print(f"{label:>10} | " + " ".join(f"{c:>12}" for c in cells))
            table.add_data(label, *cells)
        run.log({f"summary_d{demb}": table})
        if (demb, "stj-q4") in acc:
            print(f"Δ(stj-q4 − softmax): " + " ".join(
                f"{np.mean(acc[(demb, 'stj-q4')][L]) - np.mean(acc[(demb, 'softmax')][L]):+7.2f}"
                for L in LENGTHS))

    run.finish()
    print("\nDONE")


if __name__ == "__main__":
    main()
