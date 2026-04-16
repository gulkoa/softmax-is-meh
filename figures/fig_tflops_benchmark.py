"""TFLOPS and % peak utilization for Stieltjes attention on A100 + H100.

Derives forward throughput from bench CSVs:
  FLOPs_fwd = (causal ? 2 : 4) * B * H * N^2 * D   (FlashAttention convention)
  TFLOPS    = FLOPs / time_s / 1e12
  %peak     = TFLOPS / peak_hw_tflops * 100

Peak bf16 tensor-core throughput used as reference:
  A100-PCIE-40GB: 312 TFLOPS
  H100-SXM5-80GB: 989 TFLOPS

Layout: 2 rows x 4 columns
  Row 0: A100 — causal abs TFLOPS | causal % peak | non-causal abs TFLOPS | non-causal % peak
  Row 1: H100 — same order

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


def plot_abs(ax, rows, hw_name: str, causal: bool, colors) -> list:
    rows_c = [r for r in rows if r["causal"] == causal]
    keys = sorted({(r["B"], r["D"], r["q"]) for r in rows_c})
    handles = []
    for i, key in enumerate(keys):
        B, D, q = key
        sub = sorted([r for r in rows_c if (r["B"], r["D"], r["q"]) == key], key=lambda r: r["N"])
        Ns = [r["N"] for r in sub]
        tri = [r["tflops_triton"] for r in sub]
        ref = [r["tflops_ref"] for r in sub]
        label = f"B={B} D={D} q={int(q)}"
        c = colors[i % len(colors)]
        h1, = ax.plot(Ns, tri, "-o", color=c, label=f"Triton {label}", markersize=3)
        h2, = ax.plot(Ns, ref, "--x", color=c, label=f"ref {label}", markersize=3, alpha=0.6)
        handles += [h1, h2]
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$N$", fontsize=8)
    ax.set_ylabel("TFLOPS", fontsize=8)
    causal_str = "causal" if causal else "non-causal"
    ax.set_title(f"{hw_name} {causal_str}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25)
    return handles


def plot_pct(ax, rows, hw_name: str, causal: bool, peak: float, colors) -> None:
    rows_c = [r for r in rows if r["causal"] == causal]
    keys = sorted({(r["B"], r["D"], r["q"]) for r in rows_c})
    for i, key in enumerate(keys):
        B, D, q = key
        sub = sorted([r for r in rows_c if (r["B"], r["D"], r["q"]) == key], key=lambda r: r["N"])
        Ns = [r["N"] for r in sub]
        tri = [r["tflops_triton"] for r in sub]
        ref = [r["tflops_ref"] for r in sub]
        c = colors[i % len(colors)]
        ax.plot(Ns, [100 * v / peak for v in tri], "-o", color=c, markersize=3)
        ax.plot(Ns, [100 * v / peak for v in ref], "--x", color=c, markersize=3, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$N$", fontsize=8)
    ax.set_ylabel(f"% peak ({peak:.0f} TF)", fontsize=8)
    causal_str = "causal" if causal else "non-causal"
    ax.set_title(f"{hw_name} {causal_str} % peak", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25)


def main() -> None:
    a100 = load(A100_CSV)
    h100 = load(H100_CSV)

    colors = plt.get_cmap("tab10").colors

    # 2 rows × 4 cols:
    #   col 0: causal abs TFLOPS
    #   col 1: causal % peak
    #   col 2: non-causal abs TFLOPS
    #   col 3: non-causal % peak
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
    fig.suptitle("Forward TFLOPS and % peak bf16 throughput (Triton solid/circles, ref dashed/crosses)",
                 fontsize=9)

    all_handles = plot_abs(axes[0, 0], a100, "A100", causal=True,  colors=colors)
    plot_pct(axes[0, 1], a100, "A100", causal=True,  peak=PEAK_TFLOPS["A100"], colors=colors)
    plot_abs(axes[0, 2], a100, "A100", causal=False, colors=colors)
    plot_pct(axes[0, 3], a100, "A100", causal=False, peak=PEAK_TFLOPS["A100"], colors=colors)

    plot_abs(axes[1, 0], h100, "H100", causal=True,  colors=colors)
    plot_pct(axes[1, 1], h100, "H100", causal=True,  peak=PEAK_TFLOPS["H100"], colors=colors)
    plot_abs(axes[1, 2], h100, "H100", causal=False, colors=colors)
    plot_pct(axes[1, 3], h100, "H100", causal=False, peak=PEAK_TFLOPS["H100"], colors=colors)

    # De-dup legend from first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen: set[str] = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uniq_h.append(h); uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
