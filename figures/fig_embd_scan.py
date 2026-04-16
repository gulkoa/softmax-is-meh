"""Figure: max retrieval accuracy vs n_embd for softmax vs stj.

Reads results/max_1layer_*initfix_ascend/accuracy_eval_seq128_arr120_first1.json
and plots a capacity-vs-accuracy curve with one line per attention type.

Usage:
    python figures/fig_embd_scan.py
    python figures/fig_embd_scan.py --show
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
OUT_DIR = Path("/users/PAS2402/alexg/softmax/thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Returns dict: tag -> {n_embd: accuracy}"""
    data = {}
    for d in sorted(RESULTS.glob("max_1layer_*initfix_ascend")):
        f = d / "accuracy_eval_seq128_arr120_first1.json"
        if not f.exists():
            continue
        m_embd = re.search(r"embd(\d+)", d.name)
        if not m_embd:
            continue
        nembd = int(m_embd.group(1))

        raw = re.sub(r"max_1layer_", "", d.name)
        raw = re.sub(r"_seq128_embd\d+_h\d+_initfix_ascend", "", raw)
        tag = raw  # e.g. "softmax" or "stieltjes_q4.0"

        acc = json.loads(f.read_text()).get("accuracy_fixed", None)
        if acc is not None:
            data.setdefault(tag, {})[nembd] = acc
    return data


# Colours and display names for the lines we care about
LINES = [
    ("softmax",          "Softmax",    "#333333", "-",  "o",  2.2),
    ("stieltjes_q2.0",   "Stj q=2",   "#4C9BE8", "--", "s",  1.6),
    ("stieltjes_q4.0",   "Stj q=4",   "#2A7DD4", "--", "^",  1.6),
    ("stieltjes_q8.0",   "Stj q=8",   "#1A5DAA", "-.", "D",  1.6),
    ("stieltjes_q16.0",  "Stj q=16",  "#FF6B2B", "-",  "*",  2.4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data = load_data()

    fig, ax = plt.subplots(figsize=(6, 4))

    for tag, label, color, ls, marker, ms in LINES:
        if tag not in data:
            continue
        pts = sorted(data[tag].items())
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                markersize=ms * 3, linewidth=1.8, label=label, zorder=3)

    # Annotate the headline gap at n_embd=16
    sm16 = data.get("softmax", {}).get(16)
    q16_16 = data.get("stieltjes_q16.0", {}).get(16)
    if sm16 is not None and q16_16 is not None:
        ax.annotate("",
                    xy=(16, q16_16), xytext=(16, sm16),
                    arrowprops=dict(arrowstyle="<->", color="#FF6B2B", lw=1.5))
        ax.text(16.8, (sm16 + q16_16) / 2, f"+{q16_16 - sm16:.2f}",
                color="#FF6B2B", fontsize=8, va="center")

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128])
    ax.set_xticklabels(["8", "16", "32", "64", "128"])
    ax.set_xlabel("Embedding dimension (n_embd)", fontsize=11)
    ax.set_ylabel("Max retrieval accuracy", fontsize=11)
    ax.set_title("STJ advantage at small embedding dimension\n"
                 "(max retrieval, 1-layer NoPE, arr=120, seq=128)", fontsize=10)
    ax.set_ylim(0.45, 1.03)
    ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = OUT_DIR / "fig_max_retrieval_embd_scan.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved {out}")
    out_png = OUT_DIR / "fig_max_retrieval_embd_scan.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved {out_png}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
