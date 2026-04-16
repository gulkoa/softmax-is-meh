"""Scan results/ for all metrics.csv files and emit one debugging plot per task.

Each figure shows val_accuracy curves for every run with that task in its
config.json. Stj runs colored by q (viridis), softmax runs in gray.

Output: thesis/figures/debug_training_curves_<task>.pdf
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
OUTDIR = Path("/users/PAS2402/alexg/softmax/thesis/figures")


def load_run(run_dir: Path):
    cfg_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    if not cfg_path.exists() or not metrics_path.exists():
        return None
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
        eps, accs = [], []
        with metrics_path.open() as f:
            for r in csv.DictReader(f):
                eps.append(int(r["epoch"]))
                accs.append(float(r["val_accuracy"]))
        if not eps:
            return None
        return {
            "name": run_dir.name,
            "task": cfg.get("task", "unknown"),
            "attn": cfg.get("attn", "unknown"),
            "q": float(cfg.get("q", 1.0)),
            "seq_len": cfg.get("seq_len", 0),
            "n_layer": cfg.get("n_layer", 6),
            "epochs": eps,
            "val_acc": accs,
        }
    except Exception as e:
        print(f"skip {run_dir.name}: {e}")
        return None


def main() -> None:
    runs = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        r = load_run(d)
        if r:
            runs.append(r)
    print(f"Loaded {len(runs)} runs")

    by_task: dict[str, list] = {}
    for r in runs:
        by_task.setdefault(r["task"], []).append(r)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    q_values = sorted({r["q"] for r in runs if r["attn"] == "stieltjes"})
    q_norm = mcolors.LogNorm(vmin=max(0.5, min(q_values) if q_values else 1),
                             vmax=max(q_values) if q_values else 64)
    cmap = cm.get_cmap("viridis")

    for task, task_runs in by_task.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for r in task_runs:
            if r["attn"] == "softmax":
                color = "gray"
                label = f"softmax L={r['n_layer']} seq={r['seq_len']}"
                lw = 1.0
                alpha = 0.7
            else:
                color = cmap(q_norm(r["q"]))
                label = f"stj q={r['q']:g} L={r['n_layer']} seq={r['seq_len']}"
                lw = 1.0
                alpha = 0.75
            ax.plot(r["epochs"], r["val_acc"], color=color, lw=lw,
                    alpha=alpha, label=label)

        ax.axhline(1.0, color="black", ls=":", lw=0.5, alpha=0.3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val accuracy")
        ax.set_title(f"Training curves — task={task} ({len(task_runs)} runs)")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        # Legend gets crowded; only label if few runs
        if len(task_runs) <= 16:
            ax.legend(fontsize=6, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False)
        else:
            # Condensed legend: one entry per (attn, q)
            seen = set()
            handles, labels = [], []
            for line, r in zip(ax.get_lines(), task_runs):
                key = (r["attn"], r["q"])
                if key in seen:
                    continue
                seen.add(key)
                handles.append(line)
                labels.append(f"{r['attn']} q={r['q']:g}" if r["attn"] == "stieltjes" else "softmax")
            ax.legend(handles, labels, fontsize=7, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False)

        out = OUTDIR / f"debug_training_curves_{task}.png"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
