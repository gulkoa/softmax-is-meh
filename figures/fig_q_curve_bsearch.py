"""Figure: binary_search accuracy vs q (Stieltjes q-sweep, fixed metric).

Reads accuracy_fixed.json from the Ascend retrain directories, produces a
PDF at thesis/figures/fig_q_curve_bsearch.pdf.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path("/users/PAS2402/alexg/softmax/softmax-is-meh")
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures/fig_q_curve_bsearch.pdf")

RUNS = {
    1.0: REPO / "results/binary_search_stieltjes_q1.0_ascend_retrain/accuracy_fixed.json",
    2.0: REPO / "results/binary_search_stieltjes_q2.0_ascend_retrain/accuracy_fixed.json",
    4.0: REPO / "results/binary_search_stieltjes_q4.0_ascend_retrain/accuracy_fixed.json",
    8.0: REPO / "results/binary_search_stieltjes_q8.0_ascend_retrain/accuracy_fixed.json",
}
SEED43 = REPO / "results/binary_search_stieltjes_q8.0_seed43_ascend_retrain/accuracy_fixed.json"
SOFTMAX = REPO / "results/binary_search_softmax_ascend_retrain/accuracy_fixed.json"


def load_acc(p: Path) -> float:
    return json.loads(p.read_text())["accuracy_fixed"]


def main() -> None:
    qs = sorted(RUNS)
    accs = [load_acc(RUNS[q]) for q in qs]
    softmax_acc = load_acc(SOFTMAX)
    seed43_acc = load_acc(SEED43)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(qs, accs, marker="o", color="tab:blue", label="Stieltjes (seed 42)", linewidth=2)
    ax.scatter([8.0], [seed43_acc], marker="s", color="tab:orange", s=40,
               label="Stieltjes q=8 (seed 43)", zorder=5)
    ax.axhline(softmax_acc, linestyle="--", color="tab:green", label=f"softmax = {softmax_acc:.3f}")
    ax.axvspan(6, 16, alpha=0.12, color="red", label="untrainable regime")

    ax.set_xscale("log", base=2)
    ax.set_xticks(qs)
    ax.set_xticklabels([str(int(q)) for q in qs])
    ax.set_xlabel("Stieltjes sharpness parameter q")
    ax.set_ylabel("accuracy (output-only, fixed metric)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
