"""LR-headroom sweep figure (thesis/experiments/2026-10-15-lr-headroom-124m.md).

Descriptive only (the record's readings Q1-Q4 are applied by
scripts/score_lr_headroom.py, not read off this figure). One column per arm
(A: Stieltjes, no norm; D: softmax + QK-norm; LS: softmax with A's learnable
capped scale, no norm), one line per learning rate, three rows sharing the
step axis:
(a) validation perplexity (every 250 steps), y-range fitted to steps
    >= 1,000 so post-warmup bumps stay visible (earlier points run off the top);
(b) the probe's M, max |score| over layers on one fixed validation sequence
    (every 20 steps; runs without probe lines are absent). Compared only
    within a column: a score means different sharpness under each mapping;
(c) gradient norm before clipping (clip 1.0), with the failure threshold 5.
A cross marks each run's first failure event after step 2,000 under the
pre-registered rule (stability_flags.flags): in (a) the first validation above
1.05x the previous one, in (c) the first gradient norm above 5.
Entropy is not plotted: the probe computes it in bf16, which inflates it
wherever |score| reaches the thousands (see the logit-growth probe record).

    python figures/fig_lr_headroom.py [NAME=LOGS ...] [--name fig_lr_headroom]

NAME is ARM@LR (e.g. A@2e-3); LOGS are comma-separated chunk logs in order
(the later chunk wins on re-run steps). With no runs given, every sweep run
in results/lr_headroom_jobs.txt with at least one log is drawn. Login-node
script (matplotlib only). Outputs to thesis/figures/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stability_flags import flags, parse  # noqa: E402

JOBS = ROOT / "results" / "lr_headroom_jobs.txt"
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures")
T0 = 2000
YFIT = 1000      # panel (a) y-range is fitted to validations from this step on
ARMS = {"A": "A: Stieltjes, no norm", "D": "D: softmax + QK-norm",
        "LS": "LS: softmax, A's scale, no norm", "C": "C: softmax, no norm",
        "B": "B: Stieltjes + QK-norm"}
LR_C = {"3e-4": "#009E73", "6e-4": "#56B4E9", "1e-3": "#0072B2",
        "2e-3": "#E69F00", "4e-3": "#D55E00", "8e-3": "#CC79A7"}   # Okabe-Ito
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 9.5,
    "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def sweep_runs():
    runs = []
    for line in JOBS.read_text().splitlines():
        if not line.strip():
            continue
        name, *jobs = line.split()
        logs = [ROOT / "results" / f"qknorm-pilot-{j}.out" for j in jobs]
        logs = [str(p) for p in logs if p.exists()]
        if logs:
            runs.append((name, ",".join(logs)))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="ARM@LR=LOG[,LOG...]")
    ap.add_argument("--name", default="fig_lr_headroom")
    a = ap.parse_args()
    runs = [tuple(r.split("=", 1)) for r in a.runs] or sweep_runs()
    if not runs:
        sys.exit("no sweep logs yet")
    arms = [k for k in ARMS if any(n.split("@")[0] == k for n, _ in runs)]
    fig, axes = plt.subplots(3, len(arms), figsize=(3.4 * len(arms), 7.0),
                             sharex=True, squeeze=False,
                             gridspec_kw={"hspace": 0.3, "wspace": 0.28})
    last, probed = 0, set()
    vrange = {}
    for name, logs in runs:
        arm, lr = name.split("@")
        col = arms.index(arm)
        gn, loss, val, probe = parse(logs)
        f = flags(gn, loss, val, probe, T0, 9500)
        c = LR_C.get(lr, INK)
        vs = sorted(val)
        axes[0, col].plot(vs, [val[s] for s in vs], color=c, lw=1.3, label=f"lr {lr}")
        late = [val[s] for s in vs if s >= YFIT]
        if late:
            lo, hi = vrange.get(col, (late[0], late[0]))
            vrange[col] = (min(lo, *late), max(hi, *late))
        if f["first_mono"] is not None:
            s = f["first_mono"]
            axes[0, col].plot([s], [val[s]], marker="x", ms=7, mew=1.8, color=c, zorder=5)
        if f["first_gn5"] is not None:
            s = f["first_gn5"]
            axes[2, col].plot([s], [gn[s]], marker="x", ms=7, mew=1.8, color=c, zorder=5)
        ps = sorted(probe)
        if ps:
            axes[1, col].plot(ps, [probe[s] for s in ps], color=c, lw=1.0)
            probed.add(col)
        gs = sorted(gn)
        axes[2, col].plot(gs, [gn[s] for s in gs], color=c, lw=0.7)
        last = max(last, max(gs, default=0))
    for col, arm in enumerate(arms):
        top, mid, bot = axes[:, col]
        top.set_title(ARMS[arm])
        top.legend(loc="upper right")
        for ax in (top, mid, bot):
            ax.set_yscale("log")
            ax.grid(True, axis="y")
            ax.axvspan(0, T0, color=GRID, alpha=0.45, lw=0)
            ax.set_xlim(0, last + 50)
        if col in vrange:
            lo, hi = vrange[col]
            top.set_ylim(lo * 0.95, hi * 1.08)
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        top.yaxis.set_major_formatter(plain)
        top.yaxis.set_minor_formatter(FuncFormatter(
            lambda v, _: f"{v:g}" if f"{v:g}"[0] in "2346" else ""))
        if col not in probed:
            mid.text(0.5, 0.5, "no probe data", transform=mid.transAxes,
                     color=MUTED, fontsize=8, ha="center", va="center")
        bot.axhline(5, color=MUTED, lw=0.7, ls="--")
        bot.set_xlabel("optimizer step")
    axes[0, 0].set_ylabel("(a) val perplexity")
    axes[1, 0].set_ylabel("(b) probe M = max |score|")
    axes[2, 0].set_ylabel("(c) grad norm (clip 1.0)")
    axes[2, 0].annotate("5: failure threshold", (last + 50, 5), xytext=(-2, 2),
                        textcoords="offset points", color=MUTED, fontsize=7,
                        ha="right", va="bottom")
    axes[0, 0].text(0.02, 0.04, "\u00d7 first failure event (rule: after step 2,000)",
                    transform=axes[0, 0].transAxes, color=MUTED, fontsize=7,
                    ha="left", va="bottom")
    axes[2, 0].text(T0 / 2, 0.04, "warmup", transform=axes[2, 0].get_xaxis_transform(),
                    color=MUTED, fontsize=7, ha="center", va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{a.name}.{ext}")
    print(f"wrote {OUT / a.name}.pdf/.png ({len(runs)} runs, arms {', '.join(arms)})")


if __name__ == "__main__":
    main()
