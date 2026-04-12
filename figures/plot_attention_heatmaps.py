"""Generate attention pattern heatmap figures for the paper.

Produces:
1. Grid of attention matrices for softmax vs stieltjes at q={1,2,4,8} for each task
2. Per-layer attention comparison (layer 0 vs layer 5)
3. Single-head detail plots for most interesting cases

Runs on login node. Outputs PDFs to thesis/figures/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
THESIS_FIGS = Path("/users/PAS2402/alexg/softmax/thesis/figures")
THESIS_FIGS.mkdir(parents=True, exist_ok=True)


def load_attn(run_dir):
    """Load sample_attn.pt — list of (1, H, T, T) tensors per layer."""
    path = RESULTS / run_dir / "analysis" / "sample_attn.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def plot_attn_grid(task, q_values, layer=0, head=0, max_T=128, out_name=None):
    """Grid: softmax | stieltjes q=1 | q=2 | q=4 | q=8
    All from the same layer and head, same task."""
    configs = [(f"{task}_softmax", "Softmax")]
    for q in q_values:
        configs.append((f"{task}_stieltjes_q{q}_v3", f"Stieltjes q={q}"))

    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, (run, title) in zip(axes, configs):
        attn = load_attn(run)
        if attn is None:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue
        if layer >= len(attn):
            ax.set_title(f"{title}\n(no layer {layer})")
            ax.axis("off")
            continue
        w = attn[layer][0, head].float().numpy()  # (T, T)
        T = min(w.shape[0], max_T)
        w = w[:T, :T]
        # Use log scale for better visibility of tails
        vmax = w.max()
        im = ax.imshow(w, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Key position", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Query position", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Attention patterns: {task.replace('_', ' ')} task, layer {layer}, head {head}",
                 fontsize=13)
    fig.tight_layout()
    out = THESIS_FIGS / (out_name or f"attn_heatmap_{task}_L{layer}H{head}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_attn_decay(task, q_values, layer=0, head=0, max_T=128, out_name=None):
    """Log-scale plot showing attention weight vs distance from peak.

    Shows the algebraic tail vs exponential decay claim visually.
    """
    configs = [(f"{task}_softmax", "Softmax")]
    for q in q_values:
        configs.append((f"{task}_stieltjes_q{q}_v3", f"Stieltjes q={q}"))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(configs)))

    for (run, title), color in zip(configs, colors):
        attn = load_attn(run)
        if attn is None or layer >= len(attn):
            continue
        w = attn[layer][0, head].float().numpy()
        T = min(w.shape[0], max_T)
        w = w[:T, :T]

        # Average weight at each distance from the diagonal (causal)
        # For query row i, look at weights on keys 0..i and measure distance
        dists = []
        vals = []
        for i in range(T):
            for j in range(i + 1):  # causal: only j <= i
                d = i - j
                dists.append(d)
                vals.append(w[i, j])
        dists = np.array(dists)
        vals = np.array(vals)

        # Bin by distance
        max_d = int(dists.max()) + 1
        bins = np.arange(max_d + 1)
        mean_by_dist = np.zeros(max_d)
        for d in range(max_d):
            mask = dists == d
            if mask.any():
                mean_by_dist[d] = vals[mask].mean()

        ax.semilogy(range(max_d), mean_by_dist + 1e-12, label=title,
                    color=color, linewidth=2)

    ax.set_xlabel("Distance from query position (|q - k|)", fontsize=11)
    ax.set_ylabel("Mean attention weight (log scale)", fontsize=11)
    ax.set_title(f"Attention decay by distance: {task.replace('_', ' ')} "
                 f"(layer {layer}, head {head})", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = THESIS_FIGS / (out_name or f"attn_decay_{task}_L{layer}H{head}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_entropy_across_layers(tasks, q_values, out_name="entropy_per_layer.pdf"):
    """Line plot: entropy vs layer index, one line per (task, attn, q)."""
    import csv
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4),
                             squeeze=False)
    axes = axes[0]
    for ax, task in zip(axes, tasks):
        # Softmax baseline
        path = RESULTS / f"{task}_softmax" / "analysis" / "entropy.csv"
        if path.exists():
            layers, ents = [], []
            with open(path) as f:
                for row in csv.DictReader(f):
                    layers.append(int(row["layer"]))
                    heads = [float(row[f"head_{h}"]) for h in range(6)]
                    ents.append(sum(heads) / len(heads))
            ax.plot(layers, ents, "k-o", label="softmax", linewidth=2)

        # Stieltjes per q
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(q_values)))
        for q, color in zip(q_values, colors):
            path = RESULTS / f"{task}_stieltjes_q{q}_v3" / "analysis" / "entropy.csv"
            if not path.exists():
                continue
            layers, ents = [], []
            with open(path) as f:
                for row in csv.DictReader(f):
                    layers.append(int(row["layer"]))
                    heads = [float(row[f"head_{h}"]) for h in range(6)]
                    # Skip NaN rows
                    if any(h != h for h in heads):  # NaN check
                        continue
                    layers.append(int(row["layer"]))
                    ents.append(sum(heads) / len(heads))
            # Dedupe (appended twice in the NaN-check path above)
            if layers:
                seen = {}
                for l, e in zip(layers, ents):
                    seen[l] = e
                sorted_layers = sorted(seen.keys())
                ax.plot(sorted_layers, [seen[l] for l in sorted_layers],
                        "-o", color=color, label=f"q={q}")

        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Mean attention entropy (nats)", fontsize=10)
        ax.set_title(task.replace("_", " "), fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = THESIS_FIGS / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=["sorting", "bfs", "binary_search", "max"])
    parser.add_argument("--q-values", nargs="+", default=["1.0", "2.0", "4.0", "8.0"])
    args = parser.parse_args()

    print("=== Generating attention pattern figures ===\n")

    print("1. Heatmap grids (layer 0, head 0):")
    for task in args.tasks:
        plot_attn_grid(task, args.q_values, layer=0, head=0)

    print("\n2. Heatmap grids (layer 5, head 0):")
    for task in args.tasks:
        plot_attn_grid(task, args.q_values, layer=5, head=0,
                       out_name=f"attn_heatmap_{task}_L5H0.pdf")

    print("\n3. Attention decay plots (layer 0):")
    for task in args.tasks:
        plot_attn_decay(task, args.q_values, layer=0, head=0)

    print("\n4. Entropy per layer (all tasks):")
    plot_entropy_across_layers(args.tasks, args.q_values)

    print(f"\nAll figures written to {THESIS_FIGS}/")


if __name__ == "__main__":
    main()
