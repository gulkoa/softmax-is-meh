"""Convergence and lambda init visualization.

Produces:
1. NR vs BS convergence: error vs iteration per q
2. Lambda init comparison: error vs iteration per init guess
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh")
FIGS = Path("/users/PAS2402/alexg/softmax/thesis/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def plot_nr_vs_bs():
    """NR vs Binary Search convergence across q values."""
    # Data in triton/convergence.csv
    rows = []
    with open(RESULTS / "triton" / "convergence.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Group by method + q
    by_method_q = {}
    for r in rows:
        method = r["method"]
        q = float(r["q"])
        iter_n = int(r["num_iter"])
        err = float(r["max_err"])
        by_method_q.setdefault((method, q), []).append((iter_n, err))

    # Plot NR vs BS for representative q values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    q_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(q_values)))

    for ax, method in zip(axes, ["NR", "BS"]):
        for q, color in zip(q_values, colors):
            data = sorted(by_method_q.get((method, q), []))
            if not data:
                continue
            iters = [d[0] for d in data]
            errs = [max(d[1], 1e-12) for d in data]
            ax.semilogy(iters, errs, marker="o", color=color,
                        label=f"q={q}", linewidth=2)
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Max relative error", fontsize=11)
        ax.set_title(f"{method} convergence", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(y=1e-6, color="red", linestyle="--", alpha=0.5,
                   label="float32 precision")

    fig.tight_layout()
    out = FIGS / "nr_vs_bs_convergence.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_lambda_init():
    """Lambda init comparison: error vs iteration per init guess."""
    rows = []
    with open(RESULTS / "results" / "lambda_init_comparison.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Focus on N=1024 for clarity. Plot one subplot per q.
    q_values = ["1.0", "2.0", "4.0", "8.0"]
    init_order = ["n^(1/q)", "n^(1/q)/2", "n^(1/q)*2",
                  "max+n^(1/2q)", "eps=1.0", "eps=0.1"]
    # Deduplicate: max+1 and eps=1.0 are identical, skip max+1
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(init_order)))

    fig, axes = plt.subplots(1, len(q_values), figsize=(4.5 * len(q_values), 5),
                             squeeze=False)
    axes = axes[0]

    for ax, q in zip(axes, q_values):
        for init_name, color in zip(init_order, colors):
            iter_to_err = {}
            for r in rows:
                if r["N"] != "1024" or r["q"] != q or r["init_name"] != init_name:
                    continue
                iter_to_err[int(r["iteration"])] = float(r["mean_rel_error"])
            if not iter_to_err:
                continue
            iters = sorted(iter_to_err.keys())
            errs = [max(iter_to_err[i], 1e-12) for i in iters]
            ax.semilogy(iters, errs, marker="o", color=color,
                        label=init_name, linewidth=2)
        ax.set_xlabel("NR iteration", fontsize=10)
        ax.set_ylabel("Mean relative error", fontsize=10)
        ax.set_title(f"q = {q}", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(y=1e-6, color="red", linestyle="--", alpha=0.5)

    fig.suptitle("Lambda initial guess convergence (N=1024)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGS / "lambda_init_convergence.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    print("=== Generating convergence figures ===\n")
    plot_nr_vs_bs()
    plot_lambda_init()
    print(f"\nAll figures in {FIGS}/")


if __name__ == "__main__":
    main()
