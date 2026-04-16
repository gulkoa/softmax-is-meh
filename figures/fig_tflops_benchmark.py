"""TFLOPS and % peak utilization for Stieltjes attention on A100 + H100.

Derives forward throughput from bench CSVs:
  FLOPs_fwd = (causal ? 2 : 4) * B * H * N^2 * D   (FlashAttention convention)
  TFLOPS    = FLOPs / time_s / 1e12
  %peak     = TFLOPS / peak_hw_tflops * 100

Peak bf16 tensor-core throughput used as reference:
  A100-PCIE-40GB: 312 TFLOPS
  H100-SXM5-80GB: 989 TFLOPS

Plots three panels per row: causal=True primary, causal=False secondary.
Rows: A100 (top), H100 (bottom).
Columns: absolute TFLOPS (left), % peak utilization (right).

Inputs:
  results/bench_triton_vs_ref_ascend_a100.csv
  results/bench_triton_vs_ref_cardinal_h100.csv

Output:
  thesis/figures/fig_tflops_benchmark.pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

A100_CSV = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_triton_vs_ref_ascend_a100.csv")
H100_CSV = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_triton_vs_ref_cardinal_h100.csv")
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures/fig_tflops_benchmark.pdf")

PEAK_TFLOPS = {"A100": 312.0, "H100": 989.0}


def flops_fwd(B: int, H: int, N: int, D: int, causal: bool) -> float:
    return (2.0 if causal else 4.0) * B * H * N * N * D


def load(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            B = int(r["B"]); H = int(r["H"]); N = int(r["N"]); D = int(r["D"])
            q = float(r["q"]); causal = r["causal"] == "True"
            t_tri = float(r["fwd_ms_triton"]) / 1e3
            t_ref = float(r["fwd_ms_ref"]) / 1e3
            flops = flops_fwd(B, H, N, D, causal)
            out.append({
                "B": B, "H": H, "N": N, "D": D, "q": q, "causal": causal,
                "tflops_triton": flops / t_tri / 1e12,
                "tflops_ref": flops / t_ref / 1e12,
            })
    return out


def plot_panel(ax_abs, ax_pct, rows, hw_name: str, causal: bool, peak: float) -> None:
    rows_c = [r for r in rows if r["causal"] == causal]
    # Group by (B, D, q) — distinct lines; x-axis is N.
    keys = sorted({(r["B"], r["D"], r["q"]) for r in rows_c})
    colors = plt.get_cmap("tab10").colors
    for i, key in enumerate(keys):
        B, D, q = key
        sub = sorted([r for r in rows_c if (r["B"], r["D"], r["q"]) == key], key=lambda r: r["N"])
        Ns = [r["N"] for r in sub]
        tri = [r["tflops_triton"] for r in sub]
        ref = [r["tflops_ref"] for r in sub]
        label = f"B={B} D={D} q={int(q)}"
        c = colors[i % len(colors)]
        ax_abs.plot(Ns, tri, "-o", color=c, label=f"Triton {label}", markersize=4)
        ax_abs.plot(Ns, ref, "--x", color=c, label=f"ref   {label}", markersize=4, alpha=0.6)
        ax_pct.plot(Ns, [100 * v / peak for v in tri], "-o", color=c, markersize=4)
        ax_pct.plot(Ns, [100 * v / peak for v in ref], "--x", color=c, markersize=4, alpha=0.6)
    ax_abs.set_xscale("log", base=2)
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel("sequence length $N$")
    ax_abs.set_ylabel("forward TFLOPS")
    ax_abs.set_title(f"{hw_name} forward, causal={causal}")
    ax_abs.grid(True, which="both", alpha=0.3)

    ax_pct.set_xscale("log", base=2)
    ax_pct.set_xlabel("sequence length $N$")
    ax_pct.set_ylabel(f"% of {hw_name} peak bf16 ({peak:.0f} TFLOPS)")
    ax_pct.set_title(f"{hw_name} fraction of peak, causal={causal}")
    ax_pct.grid(True, which="both", alpha=0.3)


def main() -> None:
    a100 = load(A100_CSV)
    h100 = load(H100_CSV)

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))

    plot_panel(axes[0, 0], axes[0, 1], a100, "A100", causal=True, peak=PEAK_TFLOPS["A100"])
    plot_panel(axes[1, 0], axes[1, 1], a100, "A100", causal=False, peak=PEAK_TFLOPS["A100"])
    plot_panel(axes[2, 0], axes[2, 1], h100, "H100", causal=True, peak=PEAK_TFLOPS["H100"])
    plot_panel(axes[3, 0], axes[3, 1], h100, "H100", causal=False, peak=PEAK_TFLOPS["H100"])

    # Legend once, top-left panel only (else it's overwhelming)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # De-dup
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uniq.append((h, l))
    fig.legend([h for h, _ in uniq], [l for _, l in uniq],
               loc="lower center", ncol=4, fontsize=7, frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.08, hspace=0.35, wspace=0.3)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
