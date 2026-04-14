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

    labels = [s[0] for s in shapes]
    a_speeds = [s[1] for s in shapes]
    h_speeds = [s[2] for s in shapes]

    xs = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([x - width / 2 for x in xs], a_speeds, width=width,
           color="tab:blue", label="A100-40GB (geomean all: 3.37×)")
    ax.bar([x + width / 2 for x in xs], h_speeds, width=width,
           color="tab:orange", label="H100-80GB (geomean all: 3.78×)")

    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("forward speedup (Triton / PyTorch-ref)")
    ax.set_title("Triton speedup: H=8, D=64, q=4, causal")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
