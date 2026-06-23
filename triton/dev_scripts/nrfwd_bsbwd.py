"""
Prototype the Triton kernel improvement in PyTorch.

The Triton kernel does:
  forward = NR-5iter from init=1.1  (good)
  backward = IFT (q * r * (dP - δ))  (BAD — causes 4pp OOD gap vs BS)

We want to change the backward to match BS PyTorch:
  dS_ij = q * r_ij * dP_ij - κ_i · δ_(j, argmax_i)
  where κ_i = q * Σ_k dP_ik * r_ik (per-row scalar)

This script builds two new mappings:
  - StieltjesNRFwdDropDeltaBwd: drop δ only (the "simple" mod). dS = q·r·dP.
  - StieltjesNRFwdBSBwd: drop δ AND add argmax-column correction. Full BS backward.

Trains MaxRetrievalModel with each at q=4 across 3 seeds and compares
against BS and NR baselines. If StieltjesNRFwdBSBwd matches BS → port
the (drop-δ + argmax-correction) backward to Triton. If StieltjesNRFwdDropDeltaBwd
also matches BS → the simpler kernel mod is enough.
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

sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")
from maxretr_bs_vs_nr import (  # noqa: E402
    StieltjesBSTransform,
    StieltjesNRTransform,
    train_max_retrieval,
    eval_accuracy,
    _set_seeds,
    _NRMappingEnum,
    run_one,
    MaxRetrievalModel,
    ProbabilitySimplexMapping,
)


# ---------------------------------------------------------------------------
# Variant A: NR forward + diagonal-only backward (no δ, no argmax correction)
# ---------------------------------------------------------------------------

class _NRFwdDropDeltaBwdFunction(torch.autograd.Function):
    """NR-5iter forward (treating λ as autograd-detached) + dS = q·r·dP backward.

    This is the simplest possible kernel mod: drop the δ correction.
    No argmax tracking.
    """

    @staticmethod
    def forward(ctx, logits, q, num_iter, eps, lam_init, dim):
        # NR forward with autograd disabled (so backward sees λ as constant,
        # mirroring the Triton kernel where λ is stored as a saved tensor).
        with torch.no_grad():
            s_max = logits.max(dim=dim, keepdim=True).values
            x = logits - s_max
            lambd = torch.full_like(s_max, lam_init)
            for _ in range(num_iter):
                diff = (lambd - x).clamp(min=eps)
                f_val = diff.pow(-q).sum(dim=dim, keepdim=True) - 1.0
                f_deriv = -q * diff.pow(-q - 1.0).sum(dim=dim, keepdim=True)
                lambd = lambd - f_val / f_deriv

        # Now compute weights using detached λ; weights still depend on logits
        # through x_max and x_i = logits - x_max (but lambda is constant)
        # We use the same path as the Triton kernel: lambda_absolute = lambd + s_max
        # and weights = (lambda_abs - logits)^(-q)
        # But for autograd, we manually drop the delta term in backward.
        diff = (lambd - x).clamp(min=eps)
        weights = diff.pow(-q)

        # Save tensors needed for backward
        # r = (λ - x)^(-q-1)
        r = diff.pow(-q - 1.0)
        ctx.save_for_backward(r, weights)
        ctx.q = q
        ctx.dim = dim
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        r, weights = ctx.saved_tensors
        q = ctx.q
        # dS = q * r * grad_output  (drop delta entirely, no argmax correction)
        grad_logits = q * r * grad_output
        return grad_logits, None, None, None, None, None


class StieltjesNRFwdDropDeltaBwd(ProbabilitySimplexMapping):
    """NR forward + diagonal-only backward."""
    def __init__(self, q=1.0, num_iter=5, eps=1e-6, lam_init=1.1):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps
        self._lam_init = lam_init

    def translate_logits(self, logits, dim, **kwargs):
        return _NRFwdDropDeltaBwdFunction.apply(
            logits, self._q, self._num_iter, self._eps, self._lam_init, dim,
        )


# ---------------------------------------------------------------------------
# Variant B: NR forward + full BS backward (drop δ + argmax correction)
# ---------------------------------------------------------------------------

class _NRFwdBSBwdFunction(torch.autograd.Function):
    """NR-5iter forward + full PyTorch-BS-equivalent backward.

    Backward: dS_ij = q * r_ij * dP_ij - κ_i * δ_(j, argmax_i)
    where κ_i = q * Σ_k dP_ik * r_ik.

    Equivalent to PyTorch's autograd through `weights = (λ - (scores - max(scores)))^(-q)`
    when λ is detached.
    """

    @staticmethod
    def forward(ctx, logits, q, num_iter, eps, lam_init, dim):
        with torch.no_grad():
            s_max_vals, s_argmax = logits.max(dim=dim, keepdim=True)
            x = logits - s_max_vals
            lambd = torch.full_like(s_max_vals, lam_init)
            for _ in range(num_iter):
                diff = (lambd - x).clamp(min=eps)
                f_val = diff.pow(-q).sum(dim=dim, keepdim=True) - 1.0
                f_deriv = -q * diff.pow(-q - 1.0).sum(dim=dim, keepdim=True)
                lambd = lambd - f_val / f_deriv

        diff = (lambd - x).clamp(min=eps)
        weights = diff.pow(-q)
        r = diff.pow(-q - 1.0)

        ctx.save_for_backward(r, weights, s_argmax)
        ctx.q = q
        ctx.dim = dim
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        r, weights, s_argmax = ctx.saved_tensors
        q = ctx.q
        dim = ctx.dim

        # Diagonal term
        grad_logits = q * r * grad_output

        # Argmax-column correction: subtract κ_i at j = argmax_i.
        # κ_i = q * Σ_k grad_output_ik * r_ik
        kappa = q * (grad_output * r).sum(dim=dim, keepdim=True)

        # Scatter -κ_i into grad_logits at column argmax_i.
        # s_argmax has shape (..., 1); kappa has shape (..., 1); both have
        # the dim collapsed to 1. We need to subtract kappa from
        # grad_logits at the argmax index along dim.
        grad_logits = grad_logits.scatter_add(dim, s_argmax, -kappa)

        return grad_logits, None, None, None, None, None


class StieltjesNRFwdBSBwd(ProbabilitySimplexMapping):
    """NR forward + full BS-style backward (drop δ + argmax correction)."""
    def __init__(self, q=1.0, num_iter=5, eps=1e-6, lam_init=1.1):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps
        self._lam_init = lam_init

    def translate_logits(self, logits, dim, **kwargs):
        return _NRFwdBSBwdFunction.apply(
            logits, self._q, self._num_iter, self._eps, self._lam_init, dim,
        )


# ---------------------------------------------------------------------------
# Backward correctness verification
# ---------------------------------------------------------------------------

def verify_backwards(device: str):
    """Compare autograd Jacobians of each variant against the canonical
    PyTorch BS autograd (which is the ground truth for 'BS gradient')."""
    print("=" * 72)
    print("Backward correctness verification")
    print("=" * 72)

    torch.manual_seed(0)
    B, T, D = 4, 16, 32
    queries = torch.randn(B, 1, D, device=device, dtype=torch.float64)
    keys = torch.randn(B, T, D, device=device, dtype=torch.float64)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * (D ** -0.5)

    for q in [1.0, 4.0, 16.0]:
        print(f"\nq = {q}")
        bs = StieltjesBSTransform(q=q, num_iter=64, eps=1e-12).to(device).to(torch.float64)
        nr = StieltjesNRTransform(q=q, num_iter=20, eps=1e-12).to(device).to(torch.float64)
        nr_dd = StieltjesNRFwdDropDeltaBwd(q=q, num_iter=20, eps=1e-12).to(device).to(torch.float64)
        nr_bs = StieltjesNRFwdBSBwd(q=q, num_iter=20, eps=1e-12).to(device).to(torch.float64)

        # Take a single row for the Jacobian
        x_single = scores[0, 0].clone().requires_grad_(True)

        def f_bs(xx): return bs.translate_logits(xx.unsqueeze(0), dim=-1).squeeze(0)
        def f_nr(xx): return nr.translate_logits(xx.unsqueeze(0), dim=-1).squeeze(0)
        def f_nrdd(xx): return nr_dd.translate_logits(xx.unsqueeze(0), dim=-1).squeeze(0)
        def f_nrbs(xx): return nr_bs.translate_logits(xx.unsqueeze(0), dim=-1).squeeze(0)

        J_bs = torch.autograd.functional.jacobian(f_bs, x_single).squeeze()
        J_nr = torch.autograd.functional.jacobian(f_nr, x_single).squeeze()
        J_nrdd = torch.autograd.functional.jacobian(f_nrdd, x_single).squeeze()
        J_nrbs = torch.autograd.functional.jacobian(f_nrbs, x_single).squeeze()

        print(f"  max|J_NR_DD  - J_BS| = {(J_nrdd - J_bs).abs().max().item():.3e}  (drop-δ vs BS)")
        print(f"  max|J_NR_BS  - J_BS| = {(J_nrbs - J_bs).abs().max().item():.3e}  (drop-δ+argmax vs BS)")
        print(f"  max|J_NR     - J_BS| = {(J_nr - J_bs).abs().max().item():.3e}  (IFT vs BS)")
        print(f"  J_BS    diag mean = {J_bs.diag().mean().item():.4e}, "
              f"off-diag mean = {(J_bs - torch.diag(J_bs.diag())).mean().item():.4e}")
        print(f"  J_NR_DD diag mean = {J_nrdd.diag().mean().item():.4e}, "
              f"off-diag mean = {(J_nrdd - torch.diag(J_nrdd.diag())).mean().item():.4e}")
        print(f"  J_NR_BS diag mean = {J_nrbs.diag().mean().item():.4e}, "
              f"off-diag mean = {(J_nrbs - torch.diag(J_nrbs.diag())).mean().item():.4e}")


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

    verify_backwards(device)
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
        "BS (PyTorch baseline)",
        mapping_enum=_NRMappingEnum(StieltjesBSTransform),
        q=args.q, **common,
    )
    nr_result = run_one(
        "NR (IFT bwd — current Triton)",
        mapping_enum=_NRMappingEnum(StieltjesNRTransform),
        q=args.q, **common,
    )
    nrdd_result = run_one(
        "NR-fwd + DropDelta bwd",
        mapping_enum=_NRMappingEnum(StieltjesNRFwdDropDeltaBwd),
        q=args.q, **common,
    )
    nrbs_result = run_one(
        "NR-fwd + BS bwd (drop-δ + argmax)",
        mapping_enum=_NRMappingEnum(StieltjesNRFwdBSBwd),
        q=args.q, **common,
    )

    out = {
        "args": vars(args),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "training_results": [bs_result, nr_result, nrdd_result, nrbs_result],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out}")

    print("\n" + "=" * 80)
    print(f"SUMMARY  q={args.q}  seed={args.seed}")
    print("=" * 80)
    print(f"\n{'L':>6s} | {'BS':>8s} | {'NR':>8s} | {'NR_DD':>8s} | {'NR_BS':>8s} | "
          f"{'DD-BS':>8s} | {'NRBS-BS':>8s}")
    print("-" * 80)
    for L in [args.id_len] + args.ood_lens:
        a = bs_result["accuracy_by_len"][str(L)]
        b = nr_result["accuracy_by_len"][str(L)]
        c = nrdd_result["accuracy_by_len"][str(L)]
        d = nrbs_result["accuracy_by_len"][str(L)]
        print(f"{L:>6d} | {a:>7.2f}% | {b:>7.2f}% | {c:>7.2f}% | {d:>7.2f}% | "
              f"{c - a:>+7.2f} | {d - a:>+7.2f}")


if __name__ == "__main__":
    main()
