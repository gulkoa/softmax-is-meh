"""Mechanism figure: heavy-tailed attention resists RL entropy collapse.
The divergence (twin regresses held-out, stilt gains) is mechanistically
an ENTROPY-COLLAPSE story: the softmax twin's policy collapses to
near-deterministic early (min gen-entropy 0.47), overfitting the
curriculum; the Stieltjes policy stays exploratory (min 1.99, ~4
throughout) and generalizes.

(Attention entropy showed only the STATIC structural difference —
softmax sharper by construction — not the RL-driven divergence; that
lives in the policy/generation entropy, plotted here.)

Login-node (wandb API). Output: thesis/figures/fig_rl_entropy_mechanism.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

STJ, SDPA = "#0072B2", "#D55E00"
STILT = {"12567803", "12567804", "12567805", "12568655"}
TWIN = {"12567814", "12567815", "12567816", "12567817", "12567818"}

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})


def pull(jobs, key):
    api = wandb.Api()
    pts = []
    for r in api.runs("gulkoa/stieltjes-flash-attn"):
        if "grpo-code" not in r.name or not any(j in r.name for j in jobs):
            continue
        for row in r.scan_history():
            if row.get(key) is not None and row.get("step") is not None:
                pts.append((row["step"], row[key]))
    pts.sort()
    return np.array(pts) if pts else np.empty((0, 2))


stj = pull(STILT, "entropy")
sd = pull(TWIN, "entropy")


def trend(a, nb=14):
    b = np.linspace(0, a[:, 0].max(), nb)
    idx = np.digitize(a[:, 0], b)
    xs = [a[idx == k, 0].mean() for k in range(1, len(b)) if (idx == k).any()]
    ys = [np.median(a[idx == k, 1]) for k in range(1, len(b)) if (idx == k).any()]
    return xs, ys


fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.plot(stj[:, 0], stj[:, 1], ".", color=STJ, ms=2.5, alpha=0.35)
ax.plot(sd[:, 0], sd[:, 1], ".", color=SDPA, ms=2.5, alpha=0.35)
ax.plot(*trend(stj), "-", color=STJ, lw=2.2, label="Stieltjes (generalizes)")
ax.plot(*trend(sd), "-", color=SDPA, lw=2.2, label="softmax twin (regresses)")
ax.axhline(stj[:, 1].min(), color=STJ, ls=":", lw=1, alpha=0.6)
ax.axhline(sd[:, 1].min(), color=SDPA, ls=":", lw=1, alpha=0.6)
ax.annotate(f"twin min {sd[:,1].min():.2f}\n(near-deterministic)",
            (sd[sd[:, 1].argmin(), 0], sd[:, 1].min()),
            (120, 0.9), color=SDPA, fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color=SDPA, lw=0.8))
ax.text(300, 4.3, f"stilt min {stj[:,1].min():.2f}\n(stays exploratory)",
        color=STJ, fontsize=7.5, ha="center")
ax.set_xlabel("GRPO step")
ax.set_ylabel("policy (generation) entropy")
ax.set_title("Heavy-tailed attention resists RL entropy collapse",
             fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.set_ylim(0, 6)
fig.tight_layout()
out = "/users/PAS2402/alexg/softmax/thesis/figures/fig_rl_entropy_mechanism.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("wrote", out)
print(f"stilt entropy min {stj[:,1].min():.2f} mean {stj[:,1].mean():.2f}; "
      f"twin min {sd[:,1].min():.2f} mean {sd[:,1].mean():.2f}")
