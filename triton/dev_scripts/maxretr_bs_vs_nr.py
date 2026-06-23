"""
Compare two Stieltjes implementations on the same max-retrieval task.

Mapping A (binary search): probability-simplex-mappings/mappings/stieltjes.py
  - 16 iterations of bisection on lambda in [eps, N^(1/q)]
  - center logits by max, clamp logits to [-50, 50]
  - default num_iter=16, eps=1e-9

Mapping B (Newton-Raphson): softmax-is-meh/triton/stieltjes_flash_attn.py
  - 5 NR iterations, constant init lambda=1.1
  - center by max
  - default eps=1e-6

Both wrapped as ProbabilitySimplexMapping; plugged into the same
MaxRetrievalModel (one-layer, one-head, linear projections, no MLP).
Trained with identical seed, data, optimizer; evaluated at ID and OOD lengths.

Also compares the two mappings' attention weights on a fixed score tensor
to quantify how different the lambda solutions are.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Local imports
PSM_ROOT = "/users/PAS2402/alexg/softmax/probability-simplex-mappings"
if PSM_ROOT not in sys.path:
    sys.path.insert(0, PSM_ROOT)
# The repo expects to be imported as `asentmax_comp.*` but the directory is
# `probability-simplex-mappings`. Patch by adding the dir as a package alias.
import importlib.util
import types

def _alias_package(pkg_name: str, real_path: str) -> None:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [real_path]
    sys.modules[pkg_name] = pkg

_alias_package("asentmax_comp", PSM_ROOT)
_alias_package("asentmax_comp.mappings", os.path.join(PSM_ROOT, "mappings"))
_alias_package("asentmax_comp.max_retrieval_architecture",
               os.path.join(PSM_ROOT, "max_retrieval_architecture"))

# Now import the real architecture
from asentmax_comp.max_retrieval_architecture.architecture import MaxRetrievalModel  # noqa: E402
from asentmax_comp.mappings.type_enum import SimplexMappingEnum  # noqa: E402
from asentmax_comp.mappings.base_cls import ProbabilitySimplexMapping  # noqa: E402

# NOTE 2026-05-12: The working tree of mappings/stieltjes.py has been
# replaced post-a19bb33 with a Newton-Raphson 3-iter implementation
# (bisection commented out). Importing from there would silently give
# the wrong thing. Inline the a19bb33 bisection here verbatim so the
# "BS" group is unambiguously the bisection version.


class StieltjesBSTransform(ProbabilitySimplexMapping):
    """Bisection Stieltjes — verbatim copy of mappings/stieltjes.py @ a19bb33."""

    def __init__(self, q: float = 1.0, num_iter: int = 16, eps: float = 1e-9):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def _line_search_bs(self, shifted_logits, dim, lb, ub):
        for _ in range(self._num_iter):
            mid = (lb + ub) / 2.0
            prob_sum = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=self._eps), -self._q),
                dim=dim,
                keepdim=True,
            ) - 1
            lb = torch.where(prob_sum > 0, mid, lb)
            ub = torch.where(prob_sum <= 0, mid, ub)
        return lb, ub

    def translate_logits(self, logits, dim, **kwargs) -> torch.Tensor:
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max

        lb = torch.full_like(x_max, self._eps)
        ub = torch.full_like(x_max, logits.shape[dim] ** (1.0 / self._q))

        lb, ub = self._line_search_bs(
            shifted_logits=x_i, dim=dim, lb=lb, ub=ub,
        )
        lambda_1 = (lb + ub) / 2.0

        return torch.pow((lambda_1 - x_i).clamp(min=self._eps), -self._q)


# ---------------------------------------------------------------------------
# Newton-Raphson Stieltjes (mirrors the triton kernel's pytorch reference)
# ---------------------------------------------------------------------------

class StieltjesNRTransform(ProbabilitySimplexMapping):
    """Stieltjes via Newton-Raphson with constant lambda init=1.1.

    Matches softmax-is-meh/triton/stieltjes_flash_attn.py:stieltjes_attention_ref
    (lines 33-90), which is the pytorch reference for the Triton kernel.
    """

    def __init__(self, q: float = 1.0, num_iter: int = 5, eps: float = 1e-6):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def translate_logits(self, logits: torch.Tensor, dim: int, **kwargs) -> torch.Tensor:
        sq = self._q
        s_max = logits.max(dim=dim, keepdim=True).values
        x = logits - s_max  # ≤ 0

        lambd = torch.full_like(s_max, 1.1)

        for _ in range(self._num_iter):
            diff = (lambd - x).clamp(min=self._eps)
            f_val = diff.pow(-sq).sum(dim=dim, keepdim=True) - 1.0
            f_deriv = -sq * diff.pow(-sq - 1.0).sum(dim=dim, keepdim=True)
            lambd = lambd - f_val / f_deriv

        diff = (lambd - x).clamp(min=self._eps)
        return diff.pow(-sq)


# ---------------------------------------------------------------------------
# Data + training (taken from probability-simplex-mappings/main.py, trimmed)
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_max_retrieval_batch(batch_size: int, seq_len: int, n_classes: int, device: str):
    priorities = torch.rand(batch_size, seq_len, device=device)
    classes = torch.randint(0, n_classes, (batch_size, seq_len), device=device)
    argmax_idx = priorities.argmax(dim=1)
    targets = classes.gather(1, argmax_idx.unsqueeze(1)).squeeze(1).long()
    priorities_t = priorities.unsqueeze(-1)
    classes_t = F.one_hot(classes, n_classes).to(dtype=torch.float32)
    items = torch.cat([priorities_t, classes_t], dim=-1)
    queries = torch.rand(batch_size, 1, device=device)
    return items, queries, targets


def train_max_retrieval(model, *, seq_len, n_classes, device, training_steps,
                        batch_size, lr, weight_decay, warmup_steps, seed=0):
    """Train deterministically: data is regenerated each step using a
    per-step seed derived from the global seed, so two runs with the same
    `seed` see exactly the same batches.
    """
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=training_steps,
        pct_start=(warmup_steps / training_steps), anneal_strategy="cos",
    )
    loss_fn = nn.CrossEntropyLoss()

    losses: List[float] = []
    pbar = tqdm(range(training_steps), desc="train", leave=False)
    for step in pbar:
        # Deterministic batch
        g = torch.Generator(device=device)
        g.manual_seed(seed * 1_000_003 + step)
        priorities = torch.rand(batch_size, seq_len, device=device, generator=g)
        classes = torch.randint(0, n_classes, (batch_size, seq_len),
                                device=device, generator=g)
        argmax_idx = priorities.argmax(dim=1)
        targets = classes.gather(1, argmax_idx.unsqueeze(1)).squeeze(1).long()
        priorities_t = priorities.unsqueeze(-1)
        classes_t = F.one_hot(classes, n_classes).to(dtype=torch.float32)
        items = torch.cat([priorities_t, classes_t], dim=-1)
        queries = torch.rand(batch_size, 1, device=device, generator=g)

        opt.zero_grad(set_to_none=True)
        logits = model(items, queries)
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        if step % 50 == 0:
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return losses


@torch.no_grad()
def eval_accuracy(model, *, seq_len, n_classes, device, eval_samples, batch_size, seed=12345):
    model.eval()
    correct, total = 0, 0
    eval_batch_size = batch_size * 2
    steps = int(np.ceil(eval_samples / eval_batch_size))
    for i in range(steps):
        g = torch.Generator(device=device)
        g.manual_seed(seed + i)
        priorities = torch.rand(eval_batch_size, seq_len, device=device, generator=g)
        classes = torch.randint(0, n_classes, (eval_batch_size, seq_len),
                                device=device, generator=g)
        argmax_idx = priorities.argmax(dim=1)
        targets = classes.gather(1, argmax_idx.unsqueeze(1)).squeeze(1).long()
        priorities_t = priorities.unsqueeze(-1)
        classes_t = F.one_hot(classes, n_classes).to(dtype=torch.float32)
        items = torch.cat([priorities_t, classes_t], dim=-1)
        queries = torch.rand(eval_batch_size, 1, device=device, generator=g)
        logits = model(items, queries)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Standalone mapping comparison on a fixed score tensor
# ---------------------------------------------------------------------------

def compare_mappings_on_fixed_input(q: float, device: str):
    """Sanity check: identical Q/K → identical scores → compare weights."""
    torch.manual_seed(0)
    B, T, D = 4, 16, 32
    queries = torch.randn(B, 1, D, device=device)
    keys = torch.randn(B, T, D, device=device)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * (D ** -0.5)

    bs = StieltjesBSTransform(q=q, num_iter=16, eps=1e-9).to(device)
    nr = StieltjesNRTransform(q=q, num_iter=5, eps=1e-6).to(device)

    w_bs = bs.translate_logits(scores, dim=-1)
    w_nr = nr.translate_logits(scores, dim=-1)

    return {
        "scores_shape": list(scores.shape),
        "q": q,
        "row_sum_bs_mean": float(w_bs.sum(dim=-1).mean().item()),
        "row_sum_bs_std":  float(w_bs.sum(dim=-1).std().item()),
        "row_sum_nr_mean": float(w_nr.sum(dim=-1).mean().item()),
        "row_sum_nr_std":  float(w_nr.sum(dim=-1).std().item()),
        "weights_max_abs_diff": float((w_bs - w_nr).abs().max().item()),
        "weights_mean_abs_diff": float((w_bs - w_nr).abs().mean().item()),
        "argmax_match_pct": float((w_bs.argmax(-1) == w_nr.argmax(-1)).float().mean().item() * 100),
    }


# ---------------------------------------------------------------------------
# Custom mapping wrapper for the architecture (so we can swap NR in)
# ---------------------------------------------------------------------------

class _NRMappingEnum:
    """Duck-type a SimplexMappingEnum-like object for our NR mapping.

    MaxRetrievalModel accesses `simplex_mapping.value` (a class).
    """
    def __init__(self, cls):
        self.value = cls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(name: str, *, mapping_enum, q: float, d_emb: int, n_classes: int,
            item_input_dim: int, training_steps: int, batch_size: int,
            lr: float, weight_decay: float, warmup_steps: int, seed: int,
            id_len: int, ood_lens: List[int], eval_samples_id: int,
            eval_samples_ood: int, device: str):
    print(f"\n=== Training: {name} (q={q}) ===")
    _set_seeds(seed)

    model = MaxRetrievalModel(
        simplex_mapping=mapping_enum,
        d_emb=d_emb,
        n_classes=n_classes,
        item_input_dim=item_input_dim,
        query_input_dim=1,
        attn_score_scale="inv_sqrt_d",
        q=q,
    ).to(device)

    t0 = time.time()
    losses = train_max_retrieval(
        model,
        seq_len=id_len,
        n_classes=n_classes,
        device=device,
        training_steps=training_steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        seed=seed,
    )
    train_time = time.time() - t0

    accs: Dict[int, float] = {}
    for i, L in enumerate([id_len] + ood_lens):
        n_eval = eval_samples_id if i == 0 else eval_samples_ood
        acc = eval_accuracy(
            model, seq_len=L, n_classes=n_classes, device=device,
            eval_samples=n_eval, batch_size=batch_size,
        )
        accs[L] = acc
        print(f"  L={L:5d}  acc={acc:.2f}%")

    return {
        "name": name,
        "q": q,
        "train_seconds": train_time,
        "final_train_loss": losses[-1] if losses else None,
        "loss_curve_every_50_steps": losses,
        "accuracy_by_len": {str(k): v for k, v in accs.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=16.0,
                        help="Stieltjes order. 16 is the value where the "
                             "discrepancy mattered most in past ablations.")
    parser.add_argument("--training-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-len", type=int, default=16)
    parser.add_argument("--ood-lens", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--eval-samples-id", type=int, default=2048)
    parser.add_argument("--eval-samples-ood", type=int, default=1024)
    parser.add_argument("--d-emb", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--out", type=str,
                        default="/users/PAS2402/alexg/softmax/softmax-is-meh/results/maxretr_bs_vs_nr.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    item_input_dim = 1 + args.n_classes  # priority + one-hot class

    # --- Fixed-input mapping comparison (no training) ---
    fixed = []
    for qval in [1.0, 4.0, 16.0]:
        fixed.append(compare_mappings_on_fixed_input(qval, device))
        print(f"[fixed] q={qval}  weights_max|Δ|={fixed[-1]['weights_max_abs_diff']:.3e}  "
              f"argmax_match={fixed[-1]['argmax_match_pct']:.1f}%")

    # --- Training comparison ---
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

    # Use inlined StieltjesBSTransform (verbatim a19bb33) NOT the working
    # tree's mappings.stieltjes — the latter has been replaced with NR-3-iter
    # post-a19bb33 (bisection commented out). SimplexMappingEnum.stieltjes
    # would silently pull the wrong implementation.
    bs_result = run_one(
        "Stieltjes-BinarySearch (a19bb33)",
        mapping_enum=_NRMappingEnum(StieltjesBSTransform),
        q=args.q,
        **common,
    )

    nr_result = run_one(
        "Stieltjes-NewtonRaphson (triton-ref)",
        mapping_enum=_NRMappingEnum(StieltjesNRTransform),
        q=args.q,
        **common,
    )

    out = {
        "args": vars(args),
        "device": device,
        "gpu_name": (torch.cuda.get_device_name(0) if device == "cuda" else None),
        "fixed_input_comparison": fixed,
        "training_results": [bs_result, nr_result],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Pretty summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"q={args.q}, steps={args.training_steps}, ID_LEN={args.id_len}")
    print(f"\n{'L':>6s} | {'BS acc':>8s} | {'NR acc':>8s} | {'Δ':>8s}")
    print("-" * 40)
    for L in [args.id_len] + args.ood_lens:
        a = bs_result["accuracy_by_len"][str(L)]
        b = nr_result["accuracy_by_len"][str(L)]
        print(f"{L:>6d} | {a:>7.2f}% | {b:>7.2f}% | {b - a:>+7.2f}")


if __name__ == "__main__":
    main()
