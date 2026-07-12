"""
SYSTEMATIC COMPARISON vs SOFTMAX — task quality at high context.

Max-retrieval in Jack's architecture_new (MLP), 3 seeds, trained at T=16,
evaluated at L in {16..8192} (up to 512x train length):

  softmax   : Jack's Softmax mapping (the baseline the paper must beat/match)
  stj qQ    : Jack's normalized Stieltjes (PyTorch bisection)
  flashn qQ : Triton kernel normalize=True (flash fwd+bwd), num_iter=8

Softmax has no q, so it runs once per seed; Stieltjes variants at q in {2,4}.
wandb-logged incrementally.
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
QS = [2.0, 4.0]
STEPS = 3000
ID_LEN = 16
LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_CLASSES, D_EMB = 10, 128
ITEM_DIM = 1 + N_CLASSES
NUM_ITER = 8   # new validated default (job 12312766)


def run_one(model, seed, device):
    train(model, seq_len=ID_LEN, n_classes=N_CLASSES, device=device,
          steps=STEPS, bs=256, lr=1e-3, wd=1e-4,
          warmup=max(1, STEPS // 10), seed=seed)
    return {L: evaluate(model, seq_len=L, n_classes=N_CLASSES, device=device,
                        samples=2048 if L == ID_LEN else 1024, bs=256)
            for L in LENGTHS}


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"maxretr-vs-softmax-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config=dict(seeds=SEEDS, qs=QS, steps=STEPS, id_len=ID_LEN,
                    lengths=LENGTHS, num_iter=NUM_ITER,
                    variants=["softmax", "stj", "flashn"]),
    )

    acc = {}  # acc[label][L] = [per-seed]

    def record(label, accs, seed):
        acc.setdefault(label, {L: [] for L in LENGTHS})
        for L in LENGTHS:
            acc[label][L].append(accs[L])
        print(f"seed={seed} {label:12s}: "
              + " ".join(f"L{L}={accs[L]:.1f}" for L in LENGTHS), flush=True)
        run.log({"seed": seed, "label": label,
                 **{f"acc/{label}/L{L}": accs[L] for L in LENGTHS}})

    for seed in SEEDS:
        # --- softmax baseline (q-free) ---
        model = build_ref(SimplexMappingEnum.softmax, device, d_emb=D_EMB,
                          n_classes=N_CLASSES, item_dim=ITEM_DIM, sq=4.0,
                          seed=seed)
        record("softmax", run_one(model, seed, device), seed)
        del model
        torch.cuda.empty_cache()

        # --- Stieltjes variants ---
        for sq in QS:
            model = build_ref(SimplexMappingEnum.stieltjes, device, d_emb=D_EMB,
                              n_classes=N_CLASSES, item_dim=ITEM_DIM, sq=sq,
                              seed=seed)
            record(f"stj-q{sq:g}", run_one(model, seed, device), seed)
            del model
            torch.cuda.empty_cache()

            model = build_flash(device, d_emb=D_EMB, n_classes=N_CLASSES,
                                item_dim=ITEM_DIM, sq=sq, num_iter=NUM_ITER,
                                seed=seed)
            record(f"flashn-q{sq:g}", run_one(model, seed, device), seed)
            del model
            torch.cuda.empty_cache()

    labels = ["softmax"] + [f"{v}-q{sq:g}" for sq in QS for v in ("stj", "flashn")]
    print(f"\n===== mean±std over seeds {SEEDS} (trained T={ID_LEN}) =====")
    table = wandb.Table(columns=["variant"] + [f"L{L}" for L in LENGTHS])
    for label in labels:
        cells = []
        for L in LENGTHS:
            v = np.array(acc[label][L])
            cells.append(f"{v.mean():6.2f}±{v.std():4.2f}")
            run.summary[f"mean/{label}/L{L}"] = float(v.mean())
        print(f"{label:>12} | " + " ".join(f"{c:>12}" for c in cells))
        table.add_data(label, *cells)
    run.log({"summary": table})

    for sq in QS:
        print(f"\nΔ(stj-q{sq:g} − softmax):    " + " ".join(
            f"{np.mean(acc[f'stj-q{sq:g}'][L]) - np.mean(acc['softmax'][L]):+7.2f}"
            for L in LENGTHS))
        print(f"Δ(flashn-q{sq:g} − softmax): " + " ".join(
            f"{np.mean(acc[f'flashn-q{sq:g}'][L]) - np.mean(acc['softmax'][L]):+7.2f}"
            for L in LENGTHS))

    run.finish()
    print("\nDONE")


if __name__ == "__main__":
    main()
