"""Task-agnostic training-curve plotter for 1-layer/1-head nanogpt sweeps.

Reads `results/<task>_1{layer,head}_*_nope_ascend/metrics.csv` and emits one PNG
per task (aggregating all seq/q/attn configs), written to
`thesis/figures/<task>_sweep_curves_ascend.png`.

Usage:
    python plot_sweep_curves.py                 # auto-detect all tasks
    python plot_sweep_curves.py --task subtle_needle_1layer max_1head
    python plot_sweep_curves.py --show-loss     # also plot train_loss panel
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

RESULTS = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"
OUT_DIR = "/users/PAS2402/alexg/softmax/thesis/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def read_metrics(path):
    eps, tls, vls, vas = [], [], [], []
    try:
        with open(path) as f:
            r = _csv.DictReader(f)
            for row in r:
                try:
                    eps.append(int(row["epoch"]))
                    tls.append(float(row["train_loss"]))
                    vls.append(float(row["val_loss"]))
                    vas.append(float(row["val_accuracy"]))
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return eps, tls, vls, vas


def parse_dir(dirname):
    """Extract task, attn, q, seq from dirname like
    'subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend'.
    Returns (task_prefix, attn, q, seq) or None.
    """
    m = re.match(
        r"(.+?)_(softmax|stieltjes(?:_q(?P<q>[\d.]+))?)_seq(?P<seq>\d+)"
        r"(?:_d\d+)?_nope_ascend$", dirname
    )
    if not m:
        return None
    task_prefix = m.group(1)
    attn_raw = m.group(2)
    if attn_raw == "softmax":
        attn, q = "softmax", None
    else:
        attn, q = "stieltjes", float(m.group("q") or 1.0)
    return task_prefix, attn, q, int(m.group("seq"))


def config_label(attn, q):
    return "softmax" if attn == "softmax" else f"stj q={q:g}"


def plot_task(task_prefix, show_loss=False):
    """Plot training curves for one task_prefix (e.g. 'subtle_needle_1layer').
    Subplots split by train_seq.
    """
    # Group dirs by train_seq
    by_seq = defaultdict(list)
    pattern = f"{RESULTS}/{task_prefix}_*_nope_ascend"
    for d in sorted(glob.glob(pattern)):
        name = os.path.basename(d)
        parsed = parse_dir(name)
        if parsed is None:
            continue
        _, attn, q, seq = parsed
        by_seq[seq].append((d, attn, q))

    if not by_seq:
        print(f"  no data for {task_prefix}")
        return

    seqs = sorted(by_seq)
    nrows = 2 if show_loss else 1
    ncols = len(seqs)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=False, sharey=False,
        squeeze=False,
    )

    for ci, s in enumerate(seqs):
        ax_acc = axes[0, ci]
        for d, attn, q in sorted(by_seq[s], key=lambda t: (t[1] == "stieltjes", t[2] or 0)):
            eps, tls, vls, vas = read_metrics(os.path.join(d, "metrics.csv"))
            if not eps:
                continue
            lbl = config_label(attn, q)
            lw = 2.0 if attn == "softmax" else 1.2
            ls = "-" if attn == "softmax" else "--"
            ax_acc.plot(eps, vas, label=lbl, linewidth=lw, linestyle=ls, alpha=0.85)
            if show_loss:
                axes[1, ci].plot(eps, tls, label=lbl, linewidth=lw, linestyle=ls, alpha=0.85)

        ax_acc.set_title(f"train_seq={s}")
        ax_acc.set_ylabel("val_accuracy")
        ax_acc.set_ylim(-0.02, 1.05)
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(fontsize=7, loc="lower right", ncol=2)
        if show_loss:
            axes[1, ci].set_xlabel("epoch")
            axes[1, ci].set_ylabel("train_loss")
            axes[1, ci].grid(alpha=0.3)
        else:
            ax_acc.set_xlabel("epoch")

    fig.suptitle(f"{task_prefix} — Ascend training curves")
    out = os.path.join(OUT_DIR, f"{task_prefix}_sweep_curves_ascend.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def discover_tasks():
    tasks = set()
    for d in glob.glob(f"{RESULTS}/*_nope_ascend"):
        name = os.path.basename(d)
        parsed = parse_dir(name)
        if parsed:
            tasks.add(parsed[0])
    return sorted(tasks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", nargs="+", default=None, help="Task prefixes")
    p.add_argument("--show-loss", action="store_true")
    args = p.parse_args()

    tasks = args.task if args.task else discover_tasks()
    print(f"Plotting {len(tasks)} task(s): {tasks}")
    for t in tasks:
        plot_task(t, show_loss=args.show_loss)


if __name__ == "__main__":
    main()
