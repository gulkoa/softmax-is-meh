"""Figure: long-context scaling — throughput and peak memory vs N.

Reads results/bench_longctx_scaling_*.csv and produces a 2-panel figure:
  Left:  throughput (TFLOPS) vs N for all providers that don't OOM
  Right: peak memory (MB) vs N for all 4 providers (OOM shown as symbol at top)

Usage:
    python figures/fig_longctx_scaling.py
    python figures/fig_longctx_scaling.py --show
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
OUT_DIR = Path("/users/PAS2402/alexg/softmax/thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual style per provider
STYLES = {
    "stieltjes_triton": dict(label="Stj Triton (ours)", color="#E8451F", ls="-",  marker="o", ms=6, lw=2.2, zorder=4),
    "flash_sdpa":       dict(label="Flash SDPA",         color="#2A7DD4", ls="-",  marker="s", ms=5, lw=1.8, zorder=3),
    "stieltjes_ref":    dict(label="Stj PyTorch ref",    color="#E8451F", ls="--", marker="^", ms=5, lw=1.4, zorder=2),
    "naive_softmax":    dict(label="Naive softmax",      color="#333333", ls="--", marker="x", ms=6, lw=1.4, zorder=2),
}

# Theoretical O(N²) memory line for annotation
def theoretical_n2_mb(N, B=1, H=8, D=64, bytes_per_elem=2):
    """Memory for a materialised N×N fp16 score matrix: B*H*N*N*bytes."""
    return B * H * N * N * bytes_per_elem / 1024**2

def theoretical_nd_mb(N, B=1, H=8, D=64, bytes_per_elem=2, passes=5):
    """O(N·d) memory: stores only Q,K,V,O tensors = 4*B*H*N*D + small overhead."""
    return passes * B * H * N * D * bytes_per_elem / 1024**2


def load_csv() -> dict[str, dict[int, dict]]:
    """Returns {provider: {N: {ms, tflops, peak_mb, oom}}}"""
    csvs = sorted(RESULTS.glob("bench_longctx_scaling_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No bench_longctx_scaling_*.csv in {RESULTS}")
    # Use latest
    path = csvs[-1]
    print(f"Reading {path}")
    data: dict[str, dict[int, dict]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            p = row["provider"]
            N = int(row["N"])
            oom = int(row["oom"])
            ms  = None if oom else float(row["fwd_ms"])
            tf  = None if oom else float(row["fwd_tflops"])
            mb  = None if oom else float(row["fwd_peak_mb"])
            data.setdefault(p, {})[N] = dict(ms=ms, tflops=tf, peak_mb=mb, oom=oom)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data = load_csv()

    # Determine GPU tag from filename
    csvs = sorted(RESULTS.glob("bench_longctx_scaling_*.csv"))
    gpu_tag = csvs[-1].stem.replace("bench_longctx_scaling_", "").upper()

    fig, (ax_tflops, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.5))

    all_N = sorted({N for pdata in data.values() for N in pdata})
    N_arr = np.array(all_N)

    for provider, style in STYLES.items():
        if provider not in data:
            continue
        pdata = data[provider]
        Ns_ok  = [N for N in all_N if N in pdata and not pdata[N]["oom"]]
        Ns_oom = [N for N in all_N if N in pdata and pdata[N]["oom"]]

        # --- throughput panel ---
        if Ns_ok:
            xs = Ns_ok
            ys = [pdata[N]["tflops"] for N in Ns_ok]
            ax_tflops.plot(xs, ys, color=style["color"], ls=style["ls"],
                           marker=style["marker"], markersize=style["ms"],
                           linewidth=style["lw"], label=style["label"], zorder=style["zorder"])
        for N in Ns_oom:
            ax_tflops.axvline(N, color=style["color"], alpha=0.2, linewidth=0.6)

        # --- memory panel ---
        if Ns_ok:
            xs = Ns_ok
            ys = [pdata[N]["peak_mb"] for N in Ns_ok]
            ax_mem.plot(xs, ys, color=style["color"], ls=style["ls"],
                        marker=style["marker"], markersize=style["ms"],
                        linewidth=style["lw"], label=style["label"], zorder=style["zorder"])
        # OOM: mark with an X at the top of plot
        for N in Ns_oom:
            ax_mem.annotate("OOM", xy=(N, 0.92), xycoords=("data", "axes fraction"),
                            color=style["color"], fontsize=6.5, ha="center", va="bottom",
                            rotation=90)

    # Theoretical memory lines
    N_theory = np.array([2**k for k in range(10, 18)])  # 1024..131072
    mem_n2 = [theoretical_n2_mb(N) for N in N_theory]
    mem_nd = [theoretical_nd_mb(N) for N in N_theory]
    ax_mem.plot(N_theory, mem_n2, color="#bbbbbb", ls=":", lw=1.2, label=r"$O(N^2)$ theory")
    ax_mem.plot(N_theory, mem_nd, color="#cccccc", ls="-.", lw=1.0, label=r"$O(Nd)$ theory")

    # 40 GB limit line
    ax_mem.axhline(40 * 1024, color="#ff4444", lw=0.8, ls="--", alpha=0.6, label="40 GB GPU limit")

    # --- formatting ---
    for ax in (ax_tflops, ax_mem):
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_N)
        ax.set_xticklabels([f"{N//1024}k" if N >= 1024 else str(N) for N in all_N],
                           fontsize=8)
        ax.set_xlabel("Sequence length $N$", fontsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.6)

    ax_tflops.set_yscale("log")
    ax_tflops.set_ylabel("Forward TFLOPS (log)", fontsize=11)
    ax_tflops.set_title(f"Throughput vs sequence length\n({gpu_tag}, B=1, H=8, D=64, q=4, causal)", fontsize=9)
    ax_tflops.legend(fontsize=8, loc="lower right")

    ax_mem.set_yscale("log")
    ax_mem.set_ylabel("Peak CUDA memory (MB)", fontsize=11)
    ax_mem.set_title(f"Memory vs sequence length\n({gpu_tag}, B=1, H=8, D=64, q=4, causal)", fontsize=9)
    ax_mem.legend(fontsize=8, loc="upper left")

    fig.tight_layout()

    for suffix in ("pdf", "png"):
        out = OUT_DIR / f"fig_longctx_scaling.{suffix}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
