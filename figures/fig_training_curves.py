"""Figure: val_accuracy vs epoch for the key nanoGPT needle training runs.

Shows softmax saturating at multiple seq_lens while stj q=4 fails to
train at long context (seq>=4096), and softmax also flatlines at
seq=16384 under the fp32 budget.

Output: thesis/figures/fig_training_curves.pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures/fig_training_curves.pdf")

RUNS = [
    # label, metrics_path, color, linestyle
    ("softmax seq=2048 (A100)",
     ROOT / "needle_softmax_q1.0_seq2048_nope_ascend/metrics.csv",
     "tab:blue", "-"),
    ("softmax seq=4096 (A100)",
     ROOT / "needle_softmax_q1.0_seq4096_nope_ascend/metrics.csv",
     "tab:cyan", "-"),
    ("softmax seq=16384 (H100, fp32)",
     ROOT / "needle_softmax_seq16384_cardinal_h100/metrics.csv",
     "tab:purple", "-"),
    ("stj q=4 seq=2048 (A100)",
     ROOT / "needle_stieltjes_q4.0_seq2048_nope_ascend/metrics.csv",
     "tab:orange", "--"),
    ("stj q=4 seq=4096 bs=1 (A100)",
     ROOT / "needle_stieltjes_q4.0_seq4096_nope_ascend/metrics.csv",
     "tab:red", "--"),
]


def load_curve(path: Path):
    eps, accs = [], []
    if not path.exists():
        return eps, accs
    with path.open() as f:
        for r in csv.DictReader(f):
            eps.append(int(r["epoch"]))
            accs.append(float(r["val_accuracy"]))
    return eps, accs


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, path, color, ls in RUNS:
        eps, accs = load_curve(path)
        if not eps:
            print(f"skip (missing): {path}")
            continue
        ax.plot(eps, accs, color=color, linestyle=ls, label=label, linewidth=1.5)
    ax.axhline(1.0, color="gray", alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy (needle task, NoPE)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="center left")
    ax.set_title("Native-training curves on needle task")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
