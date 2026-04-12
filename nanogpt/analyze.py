"""
Attention analysis script for nanoGPT checkpoints.

Extracts attention patterns, computes entropy and concentration statistics,
and saves sample attention tensors for heatmap generation.

Usage:
    python analyze.py --checkpoint model.pt --task sorting --attn softmax --out results/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model import GPTConfig, GPT
from data import CLRSDataset, TaskConfig, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Attention weight extraction helpers
# ---------------------------------------------------------------------------

def _compute_stieltjes_weights(
    scores: torch.Tensor,
    causal_mask: torch.Tensor,
    sq: float,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Stieltjes attention normalization.

    Args:
        scores:      (B, H, T, T) raw QK^T * scale, with -inf already filled for
                     positions beyond the sequence (not needed here — causal_mask
                     is applied at the end).
        causal_mask: (T, T) boolean mask — True where we should KEEP the position
                     (i.e. lower-left triangle, diagonal inclusive).
        sq:          Stieltjes q parameter.

    Returns:
        weights: (B, H, T, T) with zeros at masked positions.
    """
    T = scores.shape[-1]
    # Mask out future positions with a very negative value so they don't affect
    # the Newton iterations.
    masked_scores = scores.masked_fill(~causal_mask, float("-inf"))

    s_max = masked_scores.max(dim=-1, keepdim=True).values
    # Replace -inf s_max (happens on the very first token row) with 0
    s_max = s_max.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    centered = masked_scores - s_max

    lambd = torch.full_like(s_max, float(T) ** (1.0 / sq))
    for _ in range(10):
        diff = (lambd - centered).clamp(min=1e-6)
        f_val = diff.pow(-sq).sum(dim=-1, keepdim=True) - 1.0
        f_deriv = -sq * diff.pow(-sq - 1.0).sum(dim=-1, keepdim=True)
        lambd = torch.maximum(lambd - f_val / f_deriv, lambd * 0.5)

    diff = (lambd - centered).clamp(min=1e-6)
    weights = diff.pow(-sq)
    weights = weights.masked_fill(~causal_mask, 0.0)
    return weights


def make_hook(
    layer_idx: int,
    attn_type: str,
    sq: float,
    storage: list,
):
    """
    Returns a forward hook for CausalSelfAttention that re-computes attention
    weights using the stored Q/K projections and saves them.

    The hook appends a tensor of shape (B, H, T, T) to storage[layer_idx].
    """

    def hook(module: nn.Module, inputs, output):
        # inputs[0] is the pre-norm x passed into the attention module
        x = inputs[0]
        B, T, C = x.shape
        n_head = module.n_head
        head_dim = module.head_dim
        sm_scale = 1.0 / math.sqrt(head_dim)

        with torch.no_grad():
            qkv = module.c_attn(x)  # (B, T, 3*C)
            q, k, _ = qkv.split(module.n_embd, dim=2)

            q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, H, T, D)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) * sm_scale  # (B, H, T, T)

            # Causal mask: True where position is valid (lower-left triangle)
            causal_mask = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=x.device)
            )  # (T, T)

            if attn_type == "softmax":
                scores_masked = scores.masked_fill(~causal_mask, float("-inf"))
                weights = torch.softmax(scores_masked, dim=-1)
                # Replace NaN rows (first token all-inf case is fine for softmax)
                weights = weights.nan_to_num(nan=0.0)
            else:  # stieltjes
                weights = _compute_stieltjes_weights(scores, causal_mask, sq)

        storage[layer_idx].append(weights.cpu())

    return hook


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Args:
        attn_weights: (B, H, T, T)
    Returns:
        (H,) mean entropy over batch and query positions
    """
    eps = 1e-10
    w = attn_weights.clamp(min=eps)
    entropy = -(w * w.log()).sum(dim=-1)  # (B, H, T)
    return entropy.mean(dim=(0, 2))  # (H,)


def compute_tail_stats(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Fraction of total weight in the top 10% of key positions.

    Args:
        attn_weights: (B, H, T, T)
    Returns:
        (H,) mean concentration over batch and query positions
    """
    B, H, T, T2 = attn_weights.shape
    sorted_w, _ = attn_weights.sort(dim=-1, descending=True)
    k = max(1, T2 // 10)
    top_k_mass = sorted_w[:, :, :, :k].sum(dim=-1)
    total_mass = sorted_w.sum(dim=-1).clamp(min=1e-10)
    concentration = (top_k_mass / total_mass).mean(dim=(0, 2))  # (H,)
    return concentration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention analysis for nanoGPT")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument(
        "--task",
        required=True,
        choices=["sorting", "binary_search", "bfs", "max"],
    )
    parser.add_argument(
        "--attn",
        required=True,
        choices=["softmax", "stieltjes"],
    )
    parser.add_argument("--q", type=float, default=1.0, help="Stieltjes q parameter")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load model from checkpoint
    # ------------------------------------------------------------------
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Support both raw state_dict checkpoints and dict-wrapped ones
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        # Try to recover config from checkpoint
        if "config" in checkpoint:
            cfg_dict = checkpoint["config"]
            if isinstance(cfg_dict, dict):
                config = GPTConfig(**cfg_dict)
            else:
                config = cfg_dict
        else:
            config = GPTConfig(
                vocab_size=VOCAB_SIZE,
                block_size=args.seq_len,
                n_layer=6,
                n_head=6,
                n_embd=384,
                attn_type=args.attn,
                stieltjes_q=args.q,
            )
    else:
        state_dict = checkpoint
        config = GPTConfig(
            vocab_size=VOCAB_SIZE,
            block_size=args.seq_len,
            n_layer=6,
            n_head=6,
            n_embd=384,
            attn_type=args.attn,
            stieltjes_q=args.q,
        )

    # Override attn settings to match CLI args
    config.attn_type = args.attn
    config.stieltjes_q = args.q

    model = GPT(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    n_layer = config.n_layer
    n_head = config.n_head

    # ------------------------------------------------------------------
    # 2. Register forward hooks to capture attention weights
    # ------------------------------------------------------------------
    # storage[layer_idx] is a list of (B, H, T, T) tensors (one per batch)
    storage: list[list[torch.Tensor]] = [[] for _ in range(n_layer)]
    hooks = []
    for layer_idx, block in enumerate(model.transformer.h):
        h = block.attn.register_forward_hook(
            make_hook(layer_idx, args.attn, args.q, storage)
        )
        hooks.append(h)

    # ------------------------------------------------------------------
    # 3. Generate data and run forward passes
    # ------------------------------------------------------------------
    task_cfg = TaskConfig(
        task_name=args.task,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
    )
    dataset = CLRSDataset(task_cfg, seed=0)

    batch_size = 32
    all_entropy: list[list[torch.Tensor]] = [[] for _ in range(n_layer)]
    all_conc: list[list[torch.Tensor]] = [[] for _ in range(n_layer)]
    sample_attn: list[torch.Tensor | None] = [None] * n_layer
    first_batch_done = False

    with torch.no_grad():
        idx = 0
        while idx < len(dataset):
            batch_end = min(idx + batch_size, len(dataset))
            xs = []
            for i in range(idx, batch_end):
                x, _ = dataset[i]
                xs.append(x)
            x_batch = torch.stack(xs).to(device)  # (B, T)

            # Clear storage from any previous batch
            for s in storage:
                s.clear()

            model(x_batch)

            # Collect stats from this batch
            for layer_idx in range(n_layer):
                if not storage[layer_idx]:
                    continue
                # storage[layer_idx] has one tensor (B, H, T, T)
                w = storage[layer_idx][0]  # (B, H, T, T)

                all_entropy[layer_idx].append(compute_entropy(w))
                all_conc[layer_idx].append(compute_tail_stats(w))

                # Save sample attention from the very first batch
                if not first_batch_done and sample_attn[layer_idx] is None:
                    sample_attn[layer_idx] = w[:1]  # (1, H, T, T)

            first_batch_done = True
            idx = batch_end

    # Remove hooks
    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------
    # 4. Aggregate statistics
    # ------------------------------------------------------------------
    # entropy[layer_idx] -> mean over all batches -> (H,)
    entropy_per_layer: list[torch.Tensor] = []
    conc_per_layer: list[torch.Tensor] = []

    for layer_idx in range(n_layer):
        if all_entropy[layer_idx]:
            entropy_per_layer.append(torch.stack(all_entropy[layer_idx]).mean(dim=0))
        else:
            entropy_per_layer.append(torch.zeros(n_head))

        if all_conc[layer_idx]:
            conc_per_layer.append(torch.stack(all_conc[layer_idx]).mean(dim=0))
        else:
            conc_per_layer.append(torch.zeros(n_head))

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    head_cols = [f"head_{i}" for i in range(n_head)]
    header = ["layer"] + head_cols

    # entropy.csv
    entropy_path = os.path.join(args.out, "entropy.csv")
    with open(entropy_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for layer_idx, ent in enumerate(entropy_per_layer):
            row = [layer_idx] + ent.tolist()
            writer.writerow(row)

    # concentration.csv
    conc_path = os.path.join(args.out, "concentration.csv")
    with open(conc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for layer_idx, conc in enumerate(conc_per_layer):
            row = [layer_idx] + conc.tolist()
            writer.writerow(row)

    # sample_attn.pt
    sample_path = os.path.join(args.out, "sample_attn.pt")
    # Replace None entries with zero tensors as a fallback
    seq_t = args.seq_len - 1  # dataset returns seq_len-1 tokens
    for layer_idx in range(n_layer):
        if sample_attn[layer_idx] is None:
            sample_attn[layer_idx] = torch.zeros(1, n_head, seq_t, seq_t)
    torch.save(sample_attn, sample_path)

    print(f"Saved entropy    -> {entropy_path}")
    print(f"Saved concentration -> {conc_path}")
    print(f"Saved sample_attn   -> {sample_path}")


if __name__ == "__main__":
    main()
