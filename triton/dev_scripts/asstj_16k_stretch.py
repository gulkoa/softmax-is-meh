"""
BUILD ON SOTA #1: AS-Stieltjes (ASEntmax-style learnable length-adaptive
scaling composed with normalized Stieltjes) on the 16k-stretch max-retrieval
protocol.

Arms: Jack's AdaptiveScalableStieltjes (scale = delta + beta(q)·(log K)^gamma,
beta per-query learnable, gamma learnable) at q_order in {4, 16}, d_emb in
{128, 256}, 3 seeds, trained T=16, eval to 16384 (1024x).

Baselines for comparison come from the scale x q sweep (job 12324697, same
protocol/seeds/eval): softmax and fixed stj-q32 tables in
thesis/findings/2026-07-13-bigger-q-and-scale-flip-the-verdict.md.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts")

from flashnorm_archnew import (  # noqa: E402
    train, evaluate, SimplexMappingEnum, MaxRetrievalModel, _set_seeds)

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

SEEDS = [0, 1, 2]
DEMBS = [int(x) for x in os.environ.get("SWEEP_DEMBS", "128,256").split(",")]
QORDERS = [float(x) for x in os.environ.get("SWEEP_QORDERS", "4,16").split(",")]
STEPS = 3000
ID_LEN = 16
LENGTHS = [16, 128, 512, 1024, 2048, 4096, 8192, 16384]
N_CLASSES = 10
ITEM_DIM = 1 + N_CLASSES


def build_asstj(device, *, d_emb, q_order, seed):
    _set_seeds(seed)
    # architecture_new auto-sets d_model=d_emb, n_heads=1 for as_stieltjes.
    return MaxRetrievalModel(
        simplex_mapping=SimplexMappingEnum.as_stieltjes, d_emb=d_emb,
        n_classes=N_CLASSES, item_input_dim=ITEM_DIM, query_input_dim=1,
        attn_score_scale="inv_sqrt_d", q_order=q_order,
    ).to(device)


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  dembs={DEMBS} qorders={QORDERS}")
    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"asstj-16k-stretch-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config=dict(seeds=SEEDS, dembs=DEMBS, q_orders=QORDERS, steps=STEPS,
                    id_len=ID_LEN, lengths=LENGTHS,
                    mapping="AdaptiveScalableStieltjes (Jack architecture_new)"),
    )

    acc = {}
    for demb in DEMBS:
        for qo in QORDERS:
            for seed in SEEDS:
                model = build_asstj(device, d_emb=demb, q_order=qo, seed=seed)
                train(model, seq_len=ID_LEN, n_classes=N_CLASSES, device=device,
                      steps=STEPS, bs=256, lr=1e-3, wd=1e-4,
                      warmup=max(1, STEPS // 10), seed=seed)
                accs = {L: evaluate(model, seq_len=L, n_classes=N_CLASSES,
                                    device=device,
                                    samples=2048 if L == ID_LEN else 1024,
                                    bs=256)
                        for L in LENGTHS}
                # log learned scaling params
                m = model._translate_logits
                gamma = float(m._log_gamma.exp().item())
                key = (demb, f"asstj-q{qo:g}")
                acc.setdefault(key, {L: [] for L in LENGTHS})
                for L in LENGTHS:
                    acc[key][L].append(accs[L])
                print(f"d={demb} q_order={qo:g} seed={seed} gamma={gamma:.3f}: "
                      + " ".join(f"L{L}={accs[L]:.1f}" for L in LENGTHS),
                      flush=True)
                run.log({"d_emb": demb, "q_order": qo, "seed": seed,
                         "gamma_learned": gamma,
                         **{f"acc/d{demb}/asstj-q{qo:g}/L{L}": accs[L]
                            for L in LENGTHS}})
                del model
                torch.cuda.empty_cache()

    print("\n===== AS-Stieltjes mean±std (compare vs scale-sweep softmax/stj-q32) =====")
    for (demb, label), d in sorted(acc.items()):
        cells = []
        for L in LENGTHS:
            v = np.array(d[L])
            cells.append(f"{v.mean():6.2f}±{v.std():4.2f}")
            run.summary[f"mean/d{demb}/{label}/L{L}"] = float(v.mean())
        print(f"d={demb} {label:>10} | " + " ".join(f"{c:>12}" for c in cells))

    run.finish()
    print("\nDONE")


if __name__ == "__main__":
    main()
