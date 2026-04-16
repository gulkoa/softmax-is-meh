"""Figure: paired A100 vs H100 forward speedup of Triton kernel vs PyTorch reference.

One panel, grouped bars per shape, A100 in blue and H100 in orange.
Covers representative training shapes at H=8, D=64, q=4 causal.

Inputs:
  results/bench_triton_vs_ref_ascend_a100.csv
  results/bench_triton_vs_ref_cardinal_h100.csv

Output:
  thesis/figures/fig_throughput_a100_vs_h100.pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

A100 = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_triton_vs_ref_ascend_a100.csv")
H100 = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_triton_vs_ref_cardinal_h100.csv")
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures/fig_throughput_a100_vs_h100.pdf")


def load_speedups(csv_path: Path) -> dict[tuple[int, int, int, int, float, str], float]:
    out = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            key = (int(r["B"]), int(r["H"]), int(r["N"]), int(r["D"]),
                   float(r["q"]), r["causal"])
            s = r["speedup_fwd"]
            if s == "nan":
                continue
            out[key] = float(s)
    return out


def main() -> None:
    a = load_speedups(A100)
    h = load_speedups(H100)

    # Shapes: H=8 D=64 q=4 causal=True, vary B and N
    shapes = []
    for N in [128, 512, 1024, 2048]:
        for B in [1, 4]:
            key = (B, 8, N, 64, 4.0, "True")
            if key in a and key in h:
                label = f"B={B}\nN={N}"
                shapes.append((label, a[key], h[key]))

    # Re-organise as 1×4 grid: one subplot per N, bars for B=1 and B=4 per GPU.
    Ns = [128, 512, 1024, 2048]
    Bs = [1, 4]
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ["tab:blue", "#6baed6", "tab:orange", "#fd8d3c"]
    bar_labels = ["A100 B=1", "A100 B=4", "H100 B=1", "H100 B=4"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    fig.suptitle("Triton forward speedup vs PyTorch ref  (H=8, D=64, q=4, causal)",
                 fontsize=9)

    for col, N in enumerate(Ns):
        ax = axes[col]
        ax.set_title(f"N = {N}", fontsize=9)
        for i, (gpu_speeds, B) in enumerate([(a, Bs[0]), (a, Bs[1]),
                                              (h, Bs[0]), (h, Bs[1])]):
            key = (B, 8, N, 64, 4.0, "True")
            val = gpu_speeds.get(key, float("nan"))
            ax.bar(offsets[i], val, width=width,
                   color=colors[i], label=bar_labels[i] if col == 0 else None)
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=1)
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel("speedup (Triton / ref)")

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
