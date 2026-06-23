"""
Controlled experiment: isolate forward vs backward as the cause of the
BS-vs-NR trained-accuracy gap.

Builds a third mapping `StieltjesBSFwdIFTBwd` whose:
  - forward is identical to a19bb33 bisection (16 iters, exact root)
  - backward uses the analytic implicit-function-theorem Jacobian
    (correct dλ/dx, matching what NR's autograd computes)

Plugs it into the same MaxRetrievalModel and trains 3 seeds at q=4.
If trained accuracy matches NR → backward is the driver of the gap.
If it matches BS → forward is the driver (current narrative is wrong).
If it lands in between → mixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Reuse the comparison script's plumbing
sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")
from maxretr_bs_vs_nr import (  # noqa: E402
    StieltjesBSTransform,
    StieltjesNRTransform,
    sample_max_retrieval_batch,
    train_max_retrieval,
    eval_accuracy,
    _set_seeds,
    _NRMappingEnum,
    run_one,
    MaxRetrievalModel,
    ProbabilitySimplexMapping,
)


# ---------------------------------------------------------------------------
# BS forward + IFT backward
# ---------------------------------------------------------------------------

class _StieltjesBSFwdIFTBwdFunction(torch.autograd.Function):
    """Forward: bisection (a19bb33). Backward: analytic IFT Jacobian.

    Forward computes λ via 16 iters of bisection (no autograd needed —
    we'll override the backward). Saves λ, x_centered, and the row-wise
    helper r = (λ-x)^{-q-1} for the backward.

    Backward uses the IFT Jacobian:
        dP_i/dx_k = q · r_i · (δ_ik − r_k / Σ_j r_j)
    So  (dL/dx_k) = Σ_i (dL/dP_i) · dP_i/dx_k
                 = q · ((dL/dP) ⊙ r) ⊗ δ_ik component
                   − q · r_k · Σ_i (dL/dP_i) · r_i / Σ_j r_j
                 = q · r_k · ( (dL/dP_k) − Σ_i (dL/dP_i)·r_i / Σ_j r_j )

    There's an additional path via x_max (logits = scores; x_centered =
    logits - logits.max). The max() gradient flows only at the argmax
    row. For simplicity we recompute x_centered = scores - scores.max
    inside the autograd Function so PyTorch handles it. Specifically,
    we accept already-clamped logits; the function then centers and
    computes weights, exposing only the weights as outputs.
    """

    @staticmethod
    def forward(ctx, logits, q, num_iter, eps, dim):
        # Match a19bb33 exactly
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max  # ≤ 0

        lb = torch.full_like(x_max, eps)
        ub = torch.full_like(x_max, logits.shape[dim] ** (1.0 / q))

        # Bisection (16 iters, with no gradient tracking on lb/ub)
        with torch.no_grad():
            for _ in range(num_iter):
                mid = (lb + ub) / 2.0
                prob_sum = torch.pow(
                    (mid - x_i).clamp(min=eps), -q
                ).sum(dim=dim, keepdim=True) - 1
                lb = torch.where(prob_sum > 0, mid, lb)
                ub = torch.where(prob_sum <= 0, mid, ub)
            lambd = (lb + ub) / 2.0

        diff = (lambd - x_i).clamp(min=eps)
        weights = diff.pow(-q)

        ctx.save_for_backward(diff, weights, lambd, x_i)
        ctx.q = q
        ctx.eps = eps
        ctx.dim = dim
        ctx.logits_shape = logits.shape
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        diff, weights, lambd, x_i = ctx.saved_tensors
        q = ctx.q
        eps = ctx.eps
        dim = ctx.dim

        # r_i = (λ - x_i)^{-q-1}
        r = diff.pow(-q - 1.0)
        # R = Σ_j r_j
        R = r.sum(dim=dim, keepdim=True).clamp(min=eps)

        # δ_i = Σ_k (grad_output_k · r_k) / R
        # dL/dx_i = q · r_i · (grad_output_i − δ_i)
        delta = (grad_output * r).sum(dim=dim, keepdim=True) / R
        grad_x_i = q * r * (grad_output - delta)

        # Chain through the max-centering: logits = x_max + x_i, where
        # x_max = logits.max. We were given grad_x_i; we want grad_logits.
        # x_i = logits - x_max; d(x_i)/d(logits_k) = δ_ik − δ_(argmax)k.
        # So grad_logits_k = grad_x_i_k − (Σ_i grad_x_i_i if k=argmax else 0).
        # For simplicity, ignore the argmax correction since logits is a
        # single tensor flowing into the rest of the model — autograd of
        # the x_max term is captured by chain rule if logits flows into
        # other places. In our setup logits only flows into this function,
        # so the missing argmax-correction term is a small shift that
        # affects the argmax row only. Both BS and NR have the same
        # max-centering path, so this term cancels in any BS-vs-NR
        # comparison. (Verify: NR also does s_max = max + center.)
        grad_logits = grad_x_i

        # Match return signature: (logits, q, num_iter, eps, dim)
        return grad_logits, None, None, None, None


class StieltjesBSFwdIFTBwd(ProbabilitySimplexMapping):
    """Forward = a19bb33 bisection; backward = analytic IFT Jacobian."""

    def __init__(self, q: float = 1.0, num_iter: int = 16, eps: float = 1e-9):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def translate_logits(self, logits, dim, **kwargs):
        return _StieltjesBSFwdIFTBwdFunction.apply(
            logits, self._q, self._num_iter, self._eps, dim,
        )


# ---------------------------------------------------------------------------
# Verification: forward should match BS exactly; backward should match NR
# ---------------------------------------------------------------------------

def verify_hybrid(device: str):
    print("=" * 72)
    print("Verifying StieltjesBSFwdIFTBwd")
    print("=" * 72)
    torch.manual_seed(0)
    B, T, D = 4, 16, 32
    queries = torch.randn(B, 1, D, device=device, dtype=torch.float64)
    keys = torch.randn(B, T, D, device=device, dtype=torch.float64)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * (D ** -0.5)

    for q in [1.0, 4.0, 16.0]:
        bs = StieltjesBSTransform(q=q, num_iter=64, eps=1e-12).to(device).to(torch.float64)
        nr = StieltjesNRTransform(q=q, num_iter=20, eps=1e-12).to(device).to(torch.float64)
        hyb = StieltjesBSFwdIFTBwd(q=q, num_iter=64, eps=1e-12).to(device).to(torch.float64)

        # Forward agreement: HYB should match BS, NR should also match BS
        # (both well-converged)
        w_bs = bs.translate_logits(scores, dim=-1)
        w_nr = nr.translate_logits(scores, dim=-1)
        w_hyb = hyb.translate_logits(scores, dim=-1)

        print(f"\nq={q}")
        print(f"  forward max|HYB - BS|  = {(w_hyb - w_bs).abs().max().item():.3e}")
        print(f"  forward max|NR  - BS|  = {(w_nr - w_bs).abs().max().item():.3e}")

        # Backward agreement: dP/dx via autograd.functional.jacobian
        x_single = scores[0, 0].clone().requires_grad_(True)

        def f_bs(xx):
            return bs.translate_logits(xx.unsqueeze(0), dim=-1).squeeze(0)
        def f_nr(xx):
            return nr.translate_logits(xx.unsqueeze(0), dim=-1).squeeze(0)
        def f_hyb(xx):
            return hyb.translate_logits(xx.unsqueeze(0), dim=-1).squeeze(0)

        J_bs = torch.autograd.functional.jacobian(f_bs, x_single).squeeze()
        J_nr = torch.autograd.functional.jacobian(f_nr, x_single).squeeze()
        J_hyb = torch.autograd.functional.jacobian(f_hyb, x_single).squeeze()

        print(f"  jacobian max|HYB - BS|  = {(J_hyb - J_bs).abs().max().item():.3e}")
        print(f"  jacobian max|HYB - NR|  = {(J_hyb - J_nr).abs().max().item():.3e}")
        print(f"  --> HYB backward should match NR (both = IFT)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--id-len", type=int, default=16)
    parser.add_argument("--ood-lens", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--eval-samples-id", type=int, default=2048)
    parser.add_argument("--eval-samples-ood", type=int, default=1024)
    parser.add_argument("--d-emb", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    verify_hybrid(device)
    if args.verify_only:
        return

    item_input_dim = 1 + args.n_classes

    common = dict(
        d_emb=args.d_emb,
        n_classes=args.n_classes,
        item_input_dim=item_input_dim,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        id_len=args.id_len,
        ood_lens=args.ood_lens,
        eval_samples_id=args.eval_samples_id,
        eval_samples_ood=args.eval_samples_ood,
        device=device,
    )

    bs_result = run_one(
        "BS (BS-fwd + BS-bwd)",
        mapping_enum=_NRMappingEnum(StieltjesBSTransform),
        q=args.q, **common,
    )
    hyb_result = run_one(
        "HYB (BS-fwd + IFT-bwd)",
        mapping_enum=_NRMappingEnum(StieltjesBSFwdIFTBwd),
        q=args.q, **common,
    )
    nr_result = run_one(
        "NR (NR-fwd + IFT-bwd)",
        mapping_enum=_NRMappingEnum(StieltjesNRTransform),
        q=args.q, **common,
    )

    out = {
        "args": vars(args),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "training_results": [bs_result, hyb_result, nr_result],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Summary
    print("\n" + "=" * 72)
    print(f"SUMMARY  q={args.q}  seed={args.seed}")
    print("=" * 72)
    print(f"\n{'L':>6s} | {'BS':>8s} | {'HYB':>8s} | {'NR':>8s} | {'HYB-BS':>8s} | {'HYB-NR':>8s}")
    print("-" * 70)
    for L in [args.id_len] + args.ood_lens:
        a = bs_result["accuracy_by_len"][str(L)]
        h = hyb_result["accuracy_by_len"][str(L)]
        n = nr_result["accuracy_by_len"][str(L)]
        print(f"{L:>6d} | {a:>7.2f}% | {h:>7.2f}% | {n:>7.2f}% | "
              f"{h - a:>+7.2f} | {h - n:>+7.2f}")


if __name__ == "__main__":
    main()
