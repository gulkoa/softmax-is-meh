"""Figure: attention entropy collapse at curriculum q=4→q=8 boundary.

Compares per-layer mean entropy on binary_search between the q=4 phase
checkpoint (epoch 25, val_acc≈0.99) and the first-q=8-epoch checkpoint
(epoch 37, val_acc≈0.10). Shows the one-epoch entropy collapse that
signals the hard-attention phase transition.

Inputs:
  results/binary_search_curriculum_q1to8_ascend/analysis_ep025_q4/entropy.csv
  results/binary_search_curriculum_q1to8_ascend/analysis_ep037_q8/entropy.csv

Output:
  thesis/figures/fig_entropy_curriculum_boundary.pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results/binary_search_curriculum_q1to8_ascend")
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures/fig_entropy_curriculum_boundary.pdf")

Q4_CSV = BASE / "analysis_ep025_q4" / "entropy.csv"
Q8_CSV = BASE / "analysis_ep037_q8" / "entropy.csv"


def load_entropy_per_layer(csv_path: Path) -> list[float]:
    """Return per-layer mean entropy (mean across heads)."""
    per_layer_means = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        n_heads = len(header) - 1
        for row in reader:
            heads = [float(x) for x in row[1:]]
            per_layer_means.append(sum(heads) / len(heads))
    return per_layer_means


def main() -> None:
    q4 = load_entropy_per_layer(Q4_CSV)
    q8 = load_entropy_per_layer(Q8_CSV)
    n_layers = len(q4)
    assert len(q8) == n_layers
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    width = 0.35
    xs = [i - width / 2 for i in layers]
    xs2 = [i + width / 2 for i in layers]
    ax.bar(xs, q4, width=width, color="tab:blue",
           label="epoch 25, q=4 (val_acc 0.99)")
    ax.bar(xs2, q8, width=width, color="tab:red",
           label="epoch 37, q=8 (val_acc 0.10)")

    ax.set_xlabel("transformer layer")
    ax.set_ylabel("mean attention entropy (nats, avg over heads)")
    ax.set_xticks(layers)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Layer-0 entropy: q=4 = {q4[0]:.3f} nats, q=8 = {q8[0]:.3f} nats")
    print(f"Mean over layers: q=4 = {sum(q4)/n_layers:.3f}, q=8 = {sum(q8)/n_layers:.3f}")


if __name__ == "__main__":
    main()
