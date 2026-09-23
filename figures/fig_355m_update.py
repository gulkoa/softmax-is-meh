"""Two figures for the lab update (Sep 2026).

(1) fig_355m_val_curves: validation perplexity vs OPTIMIZER STEP for every
    355M pretraining run (the axis the "10M tokens/step" confusion was
    about — wandb's default `_step` was the log-call index). Three panels by
    positional scheme; hue = attention (Stieltjes blue / softmax vermilion),
    dashed = + QK-norm, thin = abandoned attempt. Data:
    results/pretrain_curves_2026-09-22.csv (wandb, audited point-for-point
    against every SLURM chunk log; finding 2026-09-15-355m-model-comparison).
(2) fig_355m_length_extrapolation: PG-19 deep-tail perplexity at 1-8x the
    training context, raw vs best test-time scaling (log-n alpha; NTK theta
    x alpha for RoPE). Eval jobs 13297095 (stj-nope), 13457490 (softmax-nope
    2e-4), 14227114 (softmax-nope + qk), 14559792 (stj-nope + qk),
    13955948 / 13967688 (rope raw / theta x alpha).

Login-node script (matplotlib only). Outputs to thesis/figures/.
"""
import csv
import collections
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

CSV = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results/pretrain_curves_2026-09-22.csv")
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures")

STJ, SDPA = "#0072B2", "#D55E00"          # Okabe-Ito blue / vermilion, project-wide
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TOK_PER_STEP, TOTAL_STEPS = 524_288, 28_610

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 9.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def spread(vals, min_gap):
    """Push label positions apart (in the given space) so no two are closer than min_gap."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    pos = [vals[i] for i in order]
    for k in range(1, len(pos)):
        if pos[k] - pos[k - 1] < min_gap:
            pos[k] = pos[k - 1] + min_gap
    out = [0.0] * len(vals)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def end_labels(ax, items, x, min_gap_log10, dx):
    """items: list of (y_value, text). Labels in the right margin, leader lines to the data end."""
    ys = [math.log10(v) for v, _ in items]
    ys_lab = spread(ys, min_gap_log10)
    for (v, txt), yl in zip(items, ys_lab):
        ax.annotate(txt, xy=(x, v), xytext=(x + dx, 10 ** yl), textcoords="data",
                    va="center", ha="left", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=0, shrinkB=2),
                    annotation_clip=False)


def load_curves():
    s = collections.defaultdict(list)
    with CSV.open() as f:
        for r in csv.DictReader(f):
            s[r["label"]].append((int(r["step"]), math.exp(float(r["val_loss"]))))
    return {k: sorted(v) for k, v in s.items()}


def fig_curves():
    S = load_curves()
    # (label, hue, linestyle, legend text, thin)
    panels = [
        ("Learned positions", [
            ("gpt2-stj-q4-medium-mix", STJ, "-", "Stieltjes, lr 6e-4", False),
            ("gpt2-sdpa-medium-mix", SDPA, "-", "softmax, lr 3e-4", False),
            ("gpt2-sdpa-medium-mix-lr0.0006-diverged", SDPA, "-", "softmax, lr 6e-4 (diverged)", True),
        ]),
        ("No positional embeddings (NoPE)", [
            ("gpt2-stj-q4-nope-nope-mix-lr0.0006", STJ, "-", "Stieltjes, 6e-4", False),
            ("gpt2-stj-q4-nope-qk-nope-mix-lr0.0006", STJ, "--", "Stieltjes + QK-norm, 6e-4", False),
            ("gpt2-sdpa-nope-nope-mix-lr0.0002", SDPA, "-", "softmax, 2e-4", False),
            ("gpt2-sdpa-nope-qk-nope-mix-lr0.0006", SDPA, "--", "softmax + QK-norm, 6e-4", False),
            ("gpt2-sdpa-nope-nope-mix-lr0.0003", SDPA, "-", "softmax, 3e-4 (diverged)", True),
        ]),
        ("RoPE", [
            ("gpt2-stj-q4-rope-mix-lr0.0006", STJ, "-", "Stieltjes, 6e-4", False),
        ]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True,
                             gridspec_kw=dict(wspace=0.42))
    for ax, (title, lines) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        finals = []
        for lab, hue, ls, text, thin in lines:
            pts = S[lab]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=hue, ls=ls, lw=1.1 if thin else 2.0, alpha=0.55 if thin else 1.0,
                    solid_capstyle="round", solid_joinstyle="round", label=text, zorder=2)
            if thin:
                # mark where the abandoned attempt leaves the frame
                over = next((s for s, p in pts if s > 3000 and p > 50), xs[-1])
                ax.annotate("diverges", xy=(over, 50), xytext=(over + 2400, 41), ha="left", va="center",
                            fontsize=7.5, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=0, shrinkB=1))
            else:
                finals.append((ys[-1], f"{ys[-1]:.2f}"))
        end_labels(ax, finals, TOTAL_STEPS, 0.034, 700)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, TOTAL_STEPS)
        ax.set_yscale("log")
        ax.set_ylim(10, 50)
        ax.yaxis.set_major_locator(FixedLocator([10, 15, 20, 30, 50]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_locator(FixedLocator([0, 10_000, 20_000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))
        ax.tick_params(length=3)
        ax.set_xlabel("optimizer step")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2 if len(lines) > 3 else 1,
                  handlelength=2.2, borderaxespad=0, columnspacing=1.2)
        top = ax.secondary_xaxis("top", functions=(lambda s: s * TOK_PER_STEP / 1e9,
                                                   lambda t: t * 1e9 / TOK_PER_STEP))
        top.spines["top"].set_visible(True)
        top.spines["top"].set_color(AXIS)
        top.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15]))
        top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}B" if v else "0"))
        top.tick_params(length=3, colors=MUTED, labelcolor=INK2, labelsize=8)
    axes[0].set_ylabel("validation perplexity (held-out mix, ctx 1024)")
    fig.text(0.5, 1.045, "355M pretraining: validation perplexity vs optimizer step",
             ha="center", va="bottom", fontsize=11, fontweight="semibold", color=INK)
    fig.text(0.5, 1.005, "same 70/15/15 FineWeb-Edu / FineMath / code mix, 15B tokens, seed 0 · "
             "one optimizer step = 524,288 tokens, 28,610 steps = 15B tokens (top axis) · "
             "solid = no auxiliary norm, dashed = + QK-norm, thin = abandoned attempt",
             ha="center", va="bottom", fontsize=8, color=INK2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_355m_val_curves.{ext}")
    plt.close(fig)


def fig_length():
    L = [1024, 2048, 4096, 8192]
    # (name, hue, linestyle, raw, best test-time scaling)
    rows = [
        ("Stieltjes NoPE", STJ, "-", [27.9, 873.4, 17408.1, 23658.9], [27.9, 28.6, 45.2, 121.1]),
        ("Stieltjes NoPE + QK-norm", STJ, "--", [27.6, 313.8, 1218.7, 1274.0], [27.6, 28.6, 52.8, 152.7]),
        ("softmax NoPE", SDPA, "-", [30.3, 675.6, 20895.2, 34118.1], [30.3, 31.1, 47.3, 142.7]),
        ("softmax NoPE + QK-norm", SDPA, "--", [27.5, 196.2, 1577.4, 9837.0], [27.5, 29.0, 59.1, 169.9]),
        ("Stieltjes RoPE", STJ, ":", [27.0, 247.0, 335.0, 341.0], [27.0, 27.0, 28.7, 83.5]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(wspace=0.5))
    titles = ["raw (no test-time scaling)", "best test-time scaling (log-n α; θ × α for RoPE)"]
    for k, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_facecolor("white")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        finals = []
        for name, hue, ls, raw, best in rows:
            ys = raw if k == 0 else best
            ax.plot(L, ys, color=hue, ls=ls, lw=2.0, marker="o", ms=6, mec="white", mew=1.5,
                    solid_capstyle="round", label=name, zorder=3)
            finals.append((ys[-1], f"{ys[-1]:,.0f}" if ys[-1] >= 100 else f"{ys[-1]:.1f}"))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(L))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1024)}×\n{int(v)}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1024 / 1.15, 8192 * 1.15)
        ax.set_yscale("log")
        if k == 0:
            ax.set_ylim(20, 50_000)
            ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 1000, 10_000]))
            gap, dx = 0.06, 8192 * 0.20
        else:
            ax.set_ylim(20, 200)
            ax.yaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200]))
            gap, dx = 0.05, 8192 * 0.20
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        end_labels(ax, finals, 8192, gap, dx)
        ax.set_title(title, loc="left")
        ax.tick_params(length=3)
        ax.set_xlabel("evaluation context (× training context 1024)")
    axes[0].set_ylabel("PG-19 deep-tail perplexity")
    axes[0].legend(loc="upper left", handlelength=2.4, borderaxespad=0.2)
    fig.text(0.5, 1.03, "355M length extrapolation: NoPE and RoPE bases, 1× to 8× the training context",
             ha="center", va="bottom", fontsize=11, fontweight="semibold", color=INK)
    fig.text(0.5, 0.995, "PG-19 test books, perplexity over the last 512 positions of each window · "
             "the raw NoPE cliff (left) is what the 873 / 676 figures were; one temperature knob removes it (right)",
             ha="center", va="bottom", fontsize=8, color=INK2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_355m_length_extrapolation.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_curves()
    fig_length()
    print("wrote", OUT / "fig_355m_val_curves.{pdf,png}", "and", OUT / "fig_355m_length_extrapolation.{pdf,png}")
