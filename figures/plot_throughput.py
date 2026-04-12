"""Generate throughput scaling figures from bench_final.csv.

Produces:
1. TFLOPS vs N for forward pass (softmax / flash_sdpa / stieltjes q=1,2,4,8)
2. TFLOPS vs N for backward pass
3. Memory ceiling plot showing softmax OOM at large N
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
FIGS = Path("/users/PAS2402/alexg/softmax/thesis/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def load_bench():
    rows = []
    with open(RESULTS / "final_benchmark.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def plot_tflops(rows, mode="fwd", d=64, causal=False, out_name=None):
    """TFLOPS vs N for all providers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect data per provider
    by_provider = {}  # provider_name -> list of (N, tflops)
    for r in rows:
        if r["mode"] != mode or int(r["D"]) != d:
            continue
        if r["causal"].lower() != str(causal).lower():
            continue
        if r["tflops"] == "OOM":
            continue
        N = int(r["N"])
        tf = float(r["tflops"])
        if r["provider"] == "softmax":
            by_provider.setdefault("naive softmax", []).append((N, tf))
        elif r["provider"] == "flash_sdpa":
            by_provider.setdefault("flash SDPA (cuDNN)", []).append((N, tf))
        elif r["provider"] == "stieltjes":
            q = r["q"]
            by_provider.setdefault(f"Stieltjes q={q}", []).append((N, tf))

    # Colors: softmax baselines in gray, stieltjes in colors
    colors = {
        "naive softmax": ("gray", ":"),
        "flash SDPA (cuDNN)": ("black", "--"),
        "Stieltjes q=1.0": ("#1f77b4", "-"),
        "Stieltjes q=2.0": ("#2ca02c", "-"),
        "Stieltjes q=4.0": ("#d62728", "-"),
        "Stieltjes q=8.0": ("#9467bd", "-"),
    }

    for label in ["naive softmax", "flash SDPA (cuDNN)",
                   "Stieltjes q=1.0", "Stieltjes q=2.0",
                   "Stieltjes q=4.0", "Stieltjes q=8.0"]:
        if label not in by_provider:
            continue
        data = sorted(by_provider[label])
        ns = [d[0] for d in data]
        tfs = [d[1] for d in data]
        color, ls = colors.get(label, ("gray", "-"))
        ax.plot(ns, tfs, marker="o", color=color, linestyle=ls,
                label=label, linewidth=2)

    ax.set_xlabel("Sequence length N", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(f"Attention throughput ({mode}, D={d}, causal={causal})", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIGS / (out_name or f"throughput_{mode}_d{d}_causal{causal}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_memory_ceiling(rows, out_name="memory_ceiling.pdf"):
    """Show where each provider OOMs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    providers = ["softmax", "flash_sdpa", "stieltjes"]
    labels = {"softmax": "naive softmax", "flash_sdpa": "flash SDPA",
              "stieltjes": "Stieltjes q=1"}
    colors = {"softmax": "gray", "flash_sdpa": "black", "stieltjes": "#1f77b4"}

    for prov in providers:
        ns_ok, ns_oom = [], []
        for r in rows:
            if r["mode"] != "fwd" or int(r["D"]) != 64 or r["causal"] != "False":
                continue
            if prov == "stieltjes" and r["q"] != "1.0":
                continue
            if r["provider"] != prov:
                continue
            N = int(r["N"])
            if r["tflops"] == "OOM":
                ns_oom.append(N)
            else:
                ns_ok.append(N)
        if ns_ok:
            ax.scatter(ns_ok, [1] * len(ns_ok), marker="o", s=80,
                       color=colors[prov], label=f"{labels[prov]} (OK)")
        if ns_oom:
            ax.scatter(ns_oom, [1] * len(ns_oom), marker="x", s=120,
                       color=colors[prov], label=f"{labels[prov]} (OOM)")

    ax.set_yticks([])
    ax.set_xlabel("Sequence length N", fontsize=11)
    ax.set_title("Memory behavior: which provider runs at each N (D=64, B=4, H=8)",
                 fontsize=12)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, which="both", axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIGS / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    rows = load_bench()
    print(f"Loaded {len(rows)} benchmark rows")

    # Forward pass plots
    for d in [64, 128]:
        for causal in [False, True]:
            plot_tflops(rows, mode="fwd", d=d, causal=causal)

    # Backward pass plots
    for d in [64, 128]:
        for causal in [False, True]:
            plot_tflops(rows, mode="bwd", d=d, causal=causal)

    # Memory ceiling
    plot_memory_ceiling(rows)

    print(f"\nAll figures in {FIGS}/")


if __name__ == "__main__":
    main()
