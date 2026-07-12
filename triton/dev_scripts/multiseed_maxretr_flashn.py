"""
PERFORMANCE: repeat the max-retrieval benchmark with the NEW normalized Triton
kernel, multi-seed, against Jack's PyTorch mappings.

Repeats the architecture_new max-retrieval comparison (previously single-seed,
job 11558881) with seeds {0,1,2} x q {2,4}: Jack stieltjes_old (unnormalized),
Jack stieltjes (normalized), FLASHn (Triton normalize=True, flash fwd+bwd).
Identical init/data per seed. Eval at L in {16,32,64,128,256,512,1024,2048}.

Reuses the validated harness from flashnorm_archnew.py. Logs per-seed
accuracies and mean±std tables to wandb.
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
LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
N_CLASSES, D_EMB = 10, 128
ITEM_DIM = 1 + N_CLASSES
NUM_ITER = 20


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"multiseed-maxretr-flashn-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config=dict(seeds=SEEDS, qs=QS, steps=STEPS, id_len=ID_LEN,
                    lengths=LENGTHS, d_emb=D_EMB, n_classes=N_CLASSES,
                    num_iter=NUM_ITER,
                    variants=["stj_old", "stj", "flashn"]),
    )

    acc = {}  # acc[(q, variant)][L] = [per-seed]
    for sq in QS:
        for seed in SEEDS:
            for variant in ["stj_old", "stj", "flashn"]:
                if variant == "flashn":
                    model = build_flash(device, d_emb=D_EMB, n_classes=N_CLASSES,
                                        item_dim=ITEM_DIM, sq=sq,
                                        num_iter=NUM_ITER, seed=seed)
                else:
                    enum = (SimplexMappingEnum.stieltjes_old if variant == "stj_old"
                            else SimplexMappingEnum.stieltjes)
                    model = build_ref(enum, device, d_emb=D_EMB,
                                      n_classes=N_CLASSES, item_dim=ITEM_DIM,
                                      sq=sq, seed=seed)
                train(model, seq_len=ID_LEN, n_classes=N_CLASSES, device=device,
                      steps=STEPS, bs=256, lr=1e-3, wd=1e-4,
                      warmup=max(1, STEPS // 10), seed=seed)
                accs = {L: evaluate(model, seq_len=L, n_classes=N_CLASSES,
                                    device=device,
                                    samples=2048 if L == ID_LEN else 1024, bs=256)
                        for L in LENGTHS}
                key = (sq, variant)
                acc.setdefault(key, {L: [] for L in LENGTHS})
                for L in LENGTHS:
                    acc[key][L].append(accs[L])
                print(f"q={sq} seed={seed} {variant:8s}: "
                      + " ".join(f"L{L}={accs[L]:.1f}" for L in LENGTHS), flush=True)
                run.log({"q": sq, "seed": seed, "variant": variant,
                         **{f"acc/{variant}/q{sq:g}/L{L}": accs[L]
                            for L in LENGTHS}})
                del model
                torch.cuda.empty_cache()

    # Summary
    for sq in QS:
        print(f"\n===== q={sq} mean±std over seeds {SEEDS} =====")
        hdr = f"{'variant':>8} | " + " ".join(f"{('L' + str(L)):>12} " for L in LENGTHS)
        print(hdr)
        table = wandb.Table(columns=["variant"] + [f"L{L}" for L in LENGTHS])
        for variant in ["stj_old", "stj", "flashn"]:
            cells = []
            for L in LENGTHS:
                v = np.array(acc[(sq, variant)][L])
                cells.append(f"{v.mean():6.2f}±{v.std():4.2f}")
                run.summary[f"mean/{variant}/q{sq:g}/L{L}"] = float(v.mean())
                run.summary[f"std/{variant}/q{sq:g}/L{L}"] = float(v.std())
            print(f"{variant:>8} | " + " ".join(f"{c:>12} " for c in cells))
            table.add_data(variant, *cells)
        run.log({f"summary_q{sq:g}": table})

        # deltas flashn - stj (should be ~0) and stj - stj_old (norm advantage)
        print(f"{'Δ(FLASHn−stj)':>14}: " + " ".join(
            f"{np.mean(acc[(sq, 'flashn')][L]) - np.mean(acc[(sq, 'stj')][L]):+6.2f} "
            for L in LENGTHS))
        print(f"{'Δ(stj−stj_old)':>14}: " + " ".join(
            f"{np.mean(acc[(sq, 'stj')][L]) - np.mean(acc[(sq, 'stj_old')][L]):+6.2f} "
            for L in LENGTHS))

    run.finish()
    print("\nDONE")


if __name__ == "__main__":
    main()
