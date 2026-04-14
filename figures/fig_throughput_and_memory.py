"""Figure: Triton vs PyTorch-ref forward speedup and peak memory.

Two panels:
(a) speedup_fwd vs N, one line per (causal, B) combo.
(b) peak_mb_triton vs peak_mb_ref vs N at B=4 H=8 D=64 q=4.0 causal=True.

Input: results/bench_triton_vs_ref_ascend_a100.csv.
Output: thesis/figures/fig_throughput_and_memory.pdf.
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

CSV = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_triton_vs_ref_ascend_a100.csv")
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures/fig_throughput_and_memory.pdf")


def load_rows() -> list[dict]:
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def to_float(s: str) -> float:
    return float("nan") if s == "nan" else float(s)


def main() -> None:
    rows = load_rows()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 3.4))

    # Panel (a): speedup vs N, grouped by (B, causal) at H=8 D=64 q=4
    groups = {
        (1, "False"): ("tab:blue", "o", "B=1 non-causal"),
        (1, "True"): ("tab:blue", "s", "B=1 causal"),
        (4, "False"): ("tab:orange", "o", "B=4 non-causal"),
        (4, "True"): ("tab:orange", "s", "B=4 causal"),
    }
    for (B, causal), (color, marker, label) in groups.items():
        xs, ys = [], []
        for r in rows:
            if int(r["H"]) != 8 or int(r["D"]) != 64 or float(r["q"]) != 4.0:
                continue
            if int(r["B"]) != B or r["causal"] != causal:
                continue
            s = to_float(r["speedup_fwd"])
            if s == s:
                xs.append(int(r["N"]))
                ys.append(s)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        if xs:
            ax_a.plot(xs, ys, marker=marker, color=color, label=label, linewidth=1.8, markersize=5)

    ax_a.axhline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax_a.set_xscale("log", base=2)
    ax_a.set_xlabel("sequence length N")
    ax_a.set_ylabel("forward speedup (Triton / PyTorch-ref)")
    ax_a.set_title("(a) Forward speedup at H=8, D=64, q=4")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=8, loc="best")

    # Panel (b): peak memory vs N at B=4 H=8 D=64 q=4 causal=True
    ns, mem_tri, mem_ref = [], [], []
    for r in rows:
        if (int(r["B"]) == 4 and int(r["H"]) == 8 and int(r["D"]) == 64
                and float(r["q"]) == 4.0 and r["causal"] == "True"):
            mt = to_float(r["peak_mb_triton"])
            mr = to_float(r["peak_mb_ref"])
            if mt == mt and mr == mr:
                ns.append(int(r["N"]))
                mem_tri.append(mt)
                mem_ref.append(mr)
    order = sorted(range(len(ns)), key=lambda i: ns[i])
    ns = [ns[i] for i in order]
    mem_tri = [mem_tri[i] for i in order]
    mem_ref = [mem_ref[i] for i in order]

    ax_b.plot(ns, mem_tri, marker="o", color="tab:blue", label="Triton", linewidth=1.8, markersize=5)
    ax_b.plot(ns, mem_ref, marker="s", color="tab:red", label="PyTorch ref", linewidth=1.8, markersize=5)

    ax_b.set_xscale("log", base=2)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("sequence length N")
    ax_b.set_ylabel("peak memory (MB)")
    ax_b.set_title("(b) Peak memory at B=4, H=8, D=64, q=4, causal")
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend(fontsize=8, loc="best")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
