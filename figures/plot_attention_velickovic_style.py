"""Attention visualizations inspired by Velickovic et al. 2024 (Figure 2, 7).

Each plot shows:
  - Rows: multiple input samples
  - Columns: items/positions in the input
  - Color: attention weight on that item
  - Color intensity shows where the head is "looking"
  - Stacking samples reveals consistency of the attention pattern

For our tasks:
  - binary_search: target at position target_idx, last input token is the query
    → we expect attention at row (last query position) to peak at target_idx
  - max: argmax at some position in the array
    → attention should peak at that position
  - needle: needle at some random position
    → attention should peak at needle position
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
FIGS = Path("/users/PAS2402/alexg/softmax/thesis/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def load_attn(run_dir):
    path = RESULTS / run_dir / "analysis" / "sample_attn.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _collect_attention_rows(attn, query_row_idx, max_items=None):
    """Extract attention weights for a specific query row across all samples.

    attn: list of (B, H, T, T) tensors per layer — but sample_attn.pt is (1, H, T, T)
    query_row_idx: which query position to look at
    Returns: (H, max_items) attention distribution at that query row
    """
    if attn is None or not attn:
        return None
    # Use layer 0 by default — first attention layer sees raw input patterns
    layer0 = attn[0]  # (1, H, T, T)
    if query_row_idx >= layer0.shape[-2]:
        query_row_idx = layer0.shape[-2] - 1
    row = layer0[0, :, query_row_idx, :].float().numpy()  # (H, T)
    if max_items is not None:
        row = row[:, :max_items]
    return row


def plot_velickovic_grid(task, q_values, query_row_idx=-1, layer=0,
                          max_items=128, out_name=None):
    """Multi-panel plot: softmax + stieltjes at various q, each showing
    attention distribution per head for a specific query position."""
    configs = [(f"{task}_softmax", "Softmax")]
    for q in q_values:
        configs.append((f"{task}_stieltjes_q{q}_v3", f"Stieltjes q={q}"))

    fig, axes = plt.subplots(1, len(configs),
                             figsize=(3.5 * len(configs), 4.5),
                             sharey=True)
    if len(configs) == 1:
        axes = [axes]

    q_idx = query_row_idx if query_row_idx >= 0 else 0  # placeholder for title

    for ax, (run, title) in zip(axes, configs):
        attn = load_attn(run)
        if attn is None or not attn:
            ax.set_title(f"{title}\n(no data)", fontsize=10)
            ax.axis("off")
            continue
        if layer >= len(attn):
            ax.set_title(f"{title}\n(no layer {layer})")
            ax.axis("off")
            continue

        layer_attn = attn[layer][0].float().numpy()  # (H, T, T)
        H, T, _ = layer_attn.shape

        # If query_row_idx is -1 or beyond T, use last valid position
        q_idx = query_row_idx if query_row_idx >= 0 else T - 1
        q_idx = min(q_idx, T - 1)

        # Extract attention distribution at query_row_idx across all heads
        # Stack heads as rows, showing attention pattern per head
        data = layer_attn[:, q_idx, :]  # (H, T)
        # Trim to max_items
        T_shown = min(T, max_items)
        data = data[:, :T_shown]

        im = ax.imshow(data, aspect="auto", cmap="Blues",
                       vmin=0, vmax=data.max() if data.max() > 0 else 1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Key position", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Attention head", fontsize=10)
        ax.set_yticks(range(H))
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Attention distribution at query position {q_idx} "
                 f"({task.replace('_', ' ')}, layer {layer})",
                 fontsize=12)
    fig.tight_layout()
    out = FIGS / (out_name or f"attn_velick_{task}_L{layer}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_velickovic_scaling(task, q_values, query_row_idx=-1, layer=0,
                             max_items=128, out_name=None):
    """Scaling-style plot: show attention pattern at multiple problem sizes
    stacked vertically. Inspired by Velickovic Fig 2.

    NOTE: we only have data at seq_len=128 so this plot is a single column;
    extend once we have ctx=4096 data.
    """
    configs = [("Softmax", f"{task}_softmax")]
    for q in q_values:
        configs.append((f"Stieltjes q={q}", f"{task}_stieltjes_q{q}_v3"))

    # Collect per-head attention rows for each config
    fig, axes = plt.subplots(1, len(configs),
                             figsize=(3 * len(configs), 4),
                             sharey=True)
    if len(configs) == 1:
        axes = [axes]

    for ax, (title, run) in zip(axes, configs):
        attn = load_attn(run)
        if attn is None:
            ax.set_title(f"{title}\n(no data)", fontsize=10)
            ax.axis("off")
            continue
        layer_attn = attn[layer][0].float().numpy()  # (H, T, T)
        H, T, _ = layer_attn.shape
        q_idx = min(max(0, query_row_idx if query_row_idx >= 0 else T - 1),
                    T - 1)
        data = layer_attn[:, q_idx, :min(T, max_items)]

        im = ax.imshow(data, aspect="auto", cmap="Blues",
                       vmin=0, vmax=data.max() if data.max() > 0 else 1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Key position", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Attention head", fontsize=10)

    fig.suptitle(f"{task.replace('_', ' ')}: attention per head "
                 f"(layer {layer}, query position {q_idx})",
                 fontsize=12)
    fig.tight_layout()
    out = FIGS / (out_name or f"attn_velick_scaling_{task}_L{layer}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_multi_layer_velickovic(task, q_values, query_row_idx=-1,
                                 max_items=128, out_name=None):
    """Show attention across all 6 layers for each config.
    Rows: layers. Columns: configs (softmax, stieltjes q=1,2,4,8).
    Cell: attention distribution per head at query_row_idx.
    """
    configs = [("Softmax", f"{task}_softmax")]
    for q in q_values:
        configs.append((f"Stj q={q}", f"{task}_stieltjes_q{q}_v3"))

    # Load attentions
    loaded = []
    for title, run in configs:
        attn = load_attn(run)
        loaded.append((title, attn))

    # Determine max layers
    n_layers = max((len(a) for _, a in loaded if a), default=0)
    if n_layers == 0:
        print(f"  skip (no data for any config of {task})")
        return

    fig, axes = plt.subplots(n_layers, len(configs),
                             figsize=(2.6 * len(configs), 1.8 * n_layers),
                             squeeze=False)

    for col_idx, (title, attn) in enumerate(loaded):
        for layer in range(n_layers):
            ax = axes[layer, col_idx]
            if attn is None or layer >= len(attn):
                ax.axis("off")
                continue
            la = attn[layer][0].float().numpy()
            H, T, _ = la.shape
            q_idx = min(max(0, query_row_idx if query_row_idx >= 0 else T - 1),
                        T - 1)
            data = la[:, q_idx, :min(T, max_items)]
            ax.imshow(data, aspect="auto", cmap="Blues",
                      vmin=0, vmax=data.max() if data.max() > 0 else 1)
            if layer == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"L{layer}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{task.replace('_', ' ')}: "
                 f"attention per head across layers (query position {q_idx})",
                 fontsize=13)
    fig.tight_layout()
    out = FIGS / (out_name or f"attn_velick_multilayer_{task}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    q_values = ["1.0", "2.0", "4.0", "8.0"]

    print("=== Velickovic-style attention visualizations ===\n")

    # For each task: single-layer per-head row-wise attention
    print("1. Single-layer (L0) per-head attention:")
    for task in ["sorting", "bfs", "binary_search", "max"]:
        plot_velickovic_grid(task, q_values, query_row_idx=-1, layer=0,
                              max_items=128)

    # Last layer (where abstract task reasoning happens)
    print("\n2. Last-layer (L5) per-head attention:")
    for task in ["sorting", "bfs", "binary_search", "max"]:
        plot_velickovic_grid(task, q_values, query_row_idx=-1, layer=5,
                              max_items=128,
                              out_name=f"attn_velick_{task}_L5.png")

    # Multi-layer view: rows=layers, cols=configs
    print("\n3. Multi-layer views (all 6 layers):")
    for task in ["sorting", "bfs", "binary_search", "max"]:
        plot_multi_layer_velickovic(task, q_values, query_row_idx=-1,
                                      max_items=128)

    print(f"\nAll figures in {FIGS}/")


if __name__ == "__main__":
    main()
