"""Figure: max retrieval accuracy vs n_embd for softmax vs stj.

Reads results/max_1layer_*initfix_ascend/accuracy_eval_seq128_arr120_first1.json
and plots a capacity-vs-accuracy curve with one line per attention type.

Seed aggregation: for each (tag, n_embd), collects every available seed
variant (base dir plus any ``_seed<N>_`` suffix), plots mean with error
bars, and scatters the individual seed values behind the mean so multi-seed
dispersion is visible on the figure.

Known training failure: the softmax base dir at n_embd = 16 converged to
0.561 (chance-level floor). That single run is excluded from the mean
with the same footnote as Table~\ref{tab:embd-scan}.

Usage:
    python figures/fig_embd_scan.py
    python figures/fig_embd_scan.py --show
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
OUT_DIR = Path("/users/PAS2402/alexg/softmax/thesis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Confirmed training-failure runs to exclude, keyed by (tag, n_embd, seed_label).
# seed_label is "base" for the un-suffixed dir, "seed1"/"seed2"/... otherwise.
FAILED_RUNS = {("softmax", 16, "base"): 0.561}

# At n_embd = 16 the sweep protocol was "three fresh seeds (seed1/2/3)"; the
# un-suffixed "base" dir at that setting predates the seeded sweep. Dropping
# it across all tags keeps the n_embd = 16 row symmetric at n = 3 seeds for
# softmax, stj q=4 and stj q=16 (matches Table "tab:embd-scan").
DROP_BASE_AT_EMBD = {16}


def load_data():
    """Returns dict: tag -> {n_embd: [acc_seed0, acc_seed1, ...]}"""
    raw_runs = {}  # (tag, nembd) -> {seed_label: acc}
    for d in sorted(RESULTS.glob("max_1layer_*initfix_ascend")):
        if " " in d.name:  # skip corrupted dir names
            continue
        f = d / "accuracy_eval_seq128_arr120_first1.json"
        if not f.exists():
            continue
        m_embd = re.search(r"embd(\d+)", d.name)
        if not m_embd:
            continue
        nembd = int(m_embd.group(1))

        raw = re.sub(r"max_1layer_", "", d.name)
        raw = re.sub(r"_seq128_embd\d+_h\d+(_seed\d+)?_initfix_ascend", "", raw)
        tag = raw  # e.g. "softmax" or "stieltjes_q4.0"

        seed_m = re.search(r"_seed(\d+)_", d.name)
        seed_label = f"seed{seed_m.group(1)}" if seed_m else "base"

        acc = json.loads(f.read_text()).get("accuracy_fixed", None)
        if acc is None:
            continue
        if (tag, nembd, seed_label) in FAILED_RUNS and \
           abs(FAILED_RUNS[(tag, nembd, seed_label)] - acc) < 1e-3:
            continue  # drop confirmed training failure
        raw_runs.setdefault((tag, nembd), {})[seed_label] = acc

    data = {}
    for (tag, nembd), seeds in raw_runs.items():
        has_seeded = any(k.startswith("seed") for k in seeds)
        if nembd in DROP_BASE_AT_EMBD and has_seeded:
            kept = [a for k, a in seeds.items() if k != "base"]
        else:
            kept = list(seeds.values())
        if kept:
            data.setdefault(tag, {})[nembd] = kept
    return data


def summarize(values):
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


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

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    for tag, label, color, ls, marker, ms in LINES:
        if tag not in data:
            continue
        pts = sorted(data[tag].items())
        xs = [x for x, _ in pts]
        means, stds = [], []
        for _, vals in pts:
            mu, sd = summarize(vals)
            means.append(mu)
            stds.append(sd)

        ax.errorbar(xs, means, yerr=stds, color=color, linestyle=ls,
                    marker=marker, markersize=ms * 3, linewidth=1.8,
                    capsize=3, capthick=1.0, elinewidth=1.0,
                    label=label, zorder=3)

        # scatter individual seeds where n >= 2, lightly transparent
        for x, vals in pts:
            if len(vals) >= 2:
                ax.scatter([x] * len(vals), vals, color=color,
                           marker="x", s=18, alpha=0.45, zorder=4,
                           linewidths=0.9)

    # Annotate the headline gap at n_embd=16 (means)
    def mean_of(tag, nembd):
        vals = data.get(tag, {}).get(nembd)
        return sum(vals) / len(vals) if vals else None

    sm16 = mean_of("softmax", 16)
    q16_16 = mean_of("stieltjes_q4.0", 16)
    if sm16 is not None and q16_16 is not None:
        ax.annotate("",
                    xy=(16, q16_16), xytext=(16, sm16),
                    arrowprops=dict(arrowstyle="<->", color="#FF6B2B", lw=1.5))
        ax.text(16.8, (sm16 + q16_16) / 2, f"+{q16_16 - sm16:.3f}",
                color="#FF6B2B", fontsize=8, va="center")

    # Mark the capacity-threshold region at n_embd = 12
    ax.axvline(12, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(12.2, 0.47, "capacity\nthreshold\n(n_embd=12)", color="#888888",
            fontsize=7, va="bottom")

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 12, 16, 32, 64, 128])
    ax.set_xticklabels(["8", "12", "16", "32", "64", "128"])
    ax.set_xlabel("Embedding dimension ($n_{\\mathrm{embd}}$)", fontsize=11)
    ax.set_ylabel("Max retrieval accuracy", fontsize=11)
    ax.set_title("STJ advantage at small embedding dimension\n"
                 "(max retrieval, 1-layer NoPE, arr=120, seq=128; "
                 "error bars = $\\pm\\sigma$, $\\times$ = individual seeds)",
                 fontsize=10)
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

    # Print the summary table so regeneration is auditable
    print("\ntag, n_embd, n_seeds, mean, std, individual_values")
    for tag, _, _, _, _, _ in LINES:
        for nembd in sorted(data.get(tag, {})):
            vals = data[tag][nembd]
            mu, sd = summarize(vals)
            v = ", ".join(f"{x:.4f}" for x in vals)
            print(f"  {tag:20s} {nembd:3d}  n={len(vals)}  "
                  f"mean={mu:.4f} std={sd:.4f}  vals=[{v}]")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
