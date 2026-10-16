"""Logit-growth probe figure (thesis/experiments/2026-10-15-logit-growth-probe.md).

124M, NoPE, no norm, lr 1e-3, seed 1: softmax C' vs Stieltjes A' re-run with
the probe on (one fixed validation sequence, every 20 steps). Three panels
sharing the step axis:
(a) max |score| over layers, relative to its own value at step 2,000 (the
    pre-registered growth ratio; raw score levels are not compared across
    mappings, since a score means different sharpness under softmax and
    Stieltjes), with the reading thresholds 1.5 and 2;
(b) the same ratio per layer (thin lines), showing where the growth sits;
(c) gradient norm before clipping (clip 1.0), with the failure threshold 5.
Entropy is not plotted: the min over layers of the min-head entropy is near 0
in both arms from step ~1,500 (each has near-one-hot heads), so it does not
separate them; per-layer entropies are in the scorer's printout.
The dotted line marks t*, softmax's first logged gradient norm > 5 after
step 2,000.

    python figures/fig_logit_probe.py [--softmax LOGS] [--stj LOGS]

LOGS = comma-separated chunk logs in order (the later chunk wins on re-run
steps). Login-node script (matplotlib only). Outputs to thesis/figures/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_logit_probe import parse  # noqa: E402

RES = "/users/PAS2402/alexg/softmax/softmax-is-meh/results/qknorm-pilot-"
SOFTMAX = f"{RES}15540142.out,{RES}15556343.out"
STJ = f"{RES}15539976.out"
OUT = Path("/users/PAS2402/alexg/softmax/thesis/figures")
T0 = 2000

STJ_C, SDPA_C = "#0072B2", "#D55E00"      # Okabe-Ito blue / vermilion, project-wide
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


def existing(paths):
    return ",".join(p for p in paths.split(",") if Path(p).exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax", default=SOFTMAX)
    ap.add_argument("--stj", default=STJ)
    ap.add_argument("--name", default="fig_logit_probe")
    a = ap.parse_args()
    runs = [("softmax", SDPA_C, parse(existing(a.softmax))),
            ("Stieltjes", STJ_C, parse(existing(a.stj)))]
    gn_soft = runs[0][2][1]
    tstar = next((s for s in sorted(gn_soft) if s > T0 and gn_soft[s] > 5), None)
    last = max(max(r[2][0]) for r in runs)

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.2), sharex=True,
                             gridspec_kw={"hspace": 0.32})
    ax_m, ax_l, ax_g = axes
    for name, col, (P, G, _) in runs:
        st = [s for s in sorted(P) if s >= T0]
        ax_m.plot(st, [P[s][0] / P[T0][0] for s in st], color=col, lw=1.4, label=name)
        nl = len(P[T0][1])
        for i in range(nl):
            r = [P[s][1][i] / P[T0][1][i] for s in st]
            ax_l.plot(st, r, color=col, lw=0.7, alpha=0.75)
            if name == "softmax" and r[-1] >= 4:   # label only a runaway layer
                ax_l.annotate(f"layer {i + 1}", (st[-1], r[-1]), xytext=(4, 0),
                              textcoords="offset points", color=col, fontsize=7.5,
                              va="center")
        gs = sorted(G)
        ax_g.plot(gs, [G[s] for s in gs], color=col, lw=0.9, label=name)
    for ax in (ax_m, ax_l):
        ax.axhline(2.0, color=MUTED, lw=0.7, ls="--")
        ax.axhline(1.5, color=MUTED, lw=0.7, ls=":")
        ax.set_yscale("log")
    xr = last + 50
    # threshold labels sit left of step 2,000, where panels (a) and (b) have no data
    ax_m.annotate("2: softmax growth bar", (T0 - 60, 2.0), xytext=(0, 2),
                  textcoords="offset points", color=MUTED, fontsize=7, ha="right", va="bottom")
    ax_m.annotate("1.5: Stieltjes no-growth bar", (T0 - 60, 1.5), xytext=(0, -2),
                  textcoords="offset points", color=MUTED, fontsize=7, ha="right", va="top")
    ax_m.set_title("(a) Max attention score over layers, relative to step 2,000 (pre-registered)")
    ax_m.set_ylabel("M(t) / M(2,000)")
    ax_l.set_title("(b) The same ratio for each of the 12 layers")
    ax_l.set_ylabel("per-layer ratio")
    ax_g.axhline(5, color=MUTED, lw=0.7, ls="--")
    ax_g.annotate("5: failure threshold", (xr, 5), xytext=(-2, 2), textcoords="offset points",
                  color=MUTED, fontsize=7, ha="right", va="bottom")
    ax_g.axvspan(0, T0, color=GRID, alpha=0.45, lw=0)
    ax_g.set_yscale("log")
    ax_g.text(T0 / 2, 0.04, "warmup", transform=ax_g.get_xaxis_transform(),
              color=MUTED, fontsize=7, ha="center", va="bottom")
    ax_g.set_title("(c) Gradient norm before clipping (clip 1.0)")
    ax_g.set_ylabel("grad norm")
    ax_g.set_xlabel("optimizer step (524,288 tokens each)")
    for ax in axes:
        ax.grid(True, axis="y")
        ax.set_xlim(0, last + 50)
        if tstar is not None:
            ax.axvline(tstar, color=INK2, lw=0.8, ls=":")
    if tstar is not None:
        ax_g.annotate(f"t* = {tstar:,}: first softmax norm > 5", (tstar, ax_g.get_ylim()[1]),
                      xytext=(4, -10), textcoords="offset points", color=INK2, fontsize=7.5)
    ax_m.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{a.name}.{ext}")
    print(f"wrote {OUT / a.name}.pdf/.png (t* = {tstar}, last probe {last})")


if __name__ == "__main__":
    main()
