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
LRS = [float(x) for x in os.environ.get("SWEEP_LRS", "1e-3").split(",")]
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


# --- V2: paper-exact ASEntmax parameterization (per-position tanh-bounded
# gamma; arXiv:2506.16640 verified 2026-07-13). Jack's class uses a GLOBAL
# scalar gamma, which destabilizes at d_emb=512 (job 12333392). ---
import math as _math  # noqa: E402

import torch.nn as _nn  # noqa: E402
import torch.nn.functional as _F  # noqa: E402

import sys as _sys  # noqa: E402
_sys.path.insert(0, "/users/PAS2402/alexg/softmax/psm-architecture-new")
from mappings.as_stieltjes import (  # noqa: E402
    _bisect_lambda, _stieltjes_from_lambda)
from mappings.base_cls import ProbabilitySimplexMapping  # noqa: E402


class ASStieltjesV2(ProbabilitySimplexMapping):
    """AS-Stieltjes with the paper-exact scale:
    beta = softplus(X w_beta), gamma = s*tanh(X w_gamma) — BOTH per-query
    learnable projections; scale = delta + beta*(log K)^gamma."""

    def __init__(self, d_model: int = 64, n_heads: int = 1,
                 gamma_bound: float = 2.0, delta: float = 1.0,
                 q_order: float = 4.0, num_iter: int = 15, eps: float = 1e-9):
        super().__init__()
        self.delta = delta
        self.q_order = q_order
        self.num_iter = num_iter
        self.eps = eps
        self.gamma_bound = gamma_bound
        self.w_beta = _nn.Parameter(torch.zeros(n_heads, d_model))
        self.w_gamma = _nn.Parameter(torch.zeros(n_heads, d_model))

    def translate_logits(self, logits, dim=-1, queries=None, **kwargs):
        if queries is None:
            raise ValueError("ASStieltjesV2 requires queries.")
        if queries.dim() == 3:
            queries = queries.unsqueeze(1)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
            squeeze_out = True
        else:
            squeeze_out = False

        K = logits.size(dim if dim >= 0 else logits.dim() + dim)
        beta = _F.softplus(torch.einsum("bhqd,hd->bhq", queries, self.w_beta))
        gamma = self.gamma_bound * torch.tanh(
            torch.einsum("bhqd,hd->bhq", queries, self.w_gamma))
        logK = _math.log(max(float(K), 2.0))
        scale = self.delta + beta * (logK ** gamma)
        scaled = scale.unsqueeze(-1) * logits

        x_max = scaled.max(dim=dim, keepdim=True).values
        shifted = scaled - x_max
        lam = _bisect_lambda(shifted, dim, self.q_order, self.num_iter, self.eps)
        if os.environ.get("AS_IFT", "0") == "1":
            # smooth implicit-function gradient: one differentiable Newton
            # step on the detached root (finding 2026-07-16 — the detached
            # form's argmax discontinuity destabilizes at sharp attention)
            lam_d = lam.detach()
            diff = (lam_d - shifted).clamp_min(self.eps)
            f_val = diff.pow(-self.q_order).sum(dim, keepdim=True) - 1.0
            f_der = (-self.q_order) * diff.pow(-self.q_order - 1.0).sum(
                dim, keepdim=True)
            lam = lam_d - f_val / f_der
        probs = _stieltjes_from_lambda(shifted, lam, dim, self.q_order, self.eps)
        return probs.squeeze(1) if squeeze_out else probs


class _V2Enum:
    value = ASStieltjesV2


def build_asstj_v2(device, *, d_emb, q_order, seed):
    _set_seeds(seed)
    return MaxRetrievalModel(
        simplex_mapping=_V2Enum(), d_emb=d_emb,
        n_classes=N_CLASSES, item_input_dim=ITEM_DIM, query_input_dim=1,
        attn_score_scale="inv_sqrt_d", q_order=q_order,
        d_model=d_emb, n_heads=1,
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
            for lr in LRS:
                for seed in SEEDS:
                    if os.environ.get("AS_V2", "0") == "1":
                        model = build_asstj_v2(device, d_emb=demb, q_order=qo,
                                               seed=seed)
                    else:
                        model = build_asstj(device, d_emb=demb, q_order=qo,
                                            seed=seed)
                    train(model, seq_len=ID_LEN, n_classes=N_CLASSES, device=device,
                          steps=STEPS, bs=256, lr=lr, wd=1e-4,
                          warmup=max(1, STEPS // 10), seed=seed)
                    accs = {L: evaluate(model, seq_len=L, n_classes=N_CLASSES,
                                        device=device,
                                        samples=2048 if L == ID_LEN else 1024,
                                        bs=256)
                            for L in LENGTHS}
                    # log learned scaling params (V2 has per-query gamma —
                    # log the mean of its bounded range instead)
                    m = model._translate_logits
                    if hasattr(m, "_log_gamma"):
                        gamma = float(m._log_gamma.exp().item())
                        tag = "asstj"
                    else:
                        gamma = float(m.w_gamma.abs().mean().item())
                        tag = "asstjV2"
                    label = (f"{tag}-q{qo:g}" if len(LRS) == 1
                             else f"{tag}-q{qo:g}-lr{lr:.0e}")
                    key = (demb, label)
                    acc.setdefault(key, {L: [] for L in LENGTHS})
                    for L in LENGTHS:
                        acc[key][L].append(accs[L])
                    print(f"d={demb} q_order={qo:g} lr={lr:.0e} seed={seed} "
                          f"gamma={gamma:.3f}: "
                          + " ".join(f"L{L}={accs[L]:.1f}" for L in LENGTHS),
                          flush=True)
                    run.log({"d_emb": demb, "q_order": qo, "seed": seed,
                             "lr": lr, "gamma_learned": gamma,
                             **{f"acc/d{demb}/{label}/L{L}": accs[L]
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
