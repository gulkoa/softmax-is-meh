"""Mechanism analysis + figure: does the softmax twin sharpen attention
harder than Stieltjes during code-RL? If so, it mechanistically ties
the held-out divergence (twin overfits/regresses, stilt stable) to the
attention mechanism — the thesis 'why'.

Pulls attn_H (attention entropy) + attn_maxp trajectories logged every
25 steps from the v2 code-RL runs (stilt vs softmax twin), aggregates
by step across chunks, compares, and plots. Login-node (wandb API).
Output: thesis/figures/fig_rl_attn_mechanism.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

STJ, SDPA = "#0072B2", "#D55E00"
STILT_JOBS = {"12567803", "12567804", "12567805", "12568655"}
TWIN_JOBS = {"12567814", "12567815", "12567816", "12567817", "12567818"}
METRIC = "attn_H_l12_mean"   # mid-layer attention entropy


def collect(job_ids):
    api = wandb.Api()
    pts = []
    for r in api.runs("gulkoa/stieltjes-flash-attn"):
        if "grpo-code" not in r.name:
            continue
        if not any(j in r.name for j in job_ids):
            continue
        for row in r.scan_history():
            if row.get(METRIC) is not None and row.get("step") is not None:
                pts.append((row["step"], row[METRIC],
                            row.get("attn_maxp_l12")))
    pts.sort()
    return np.array(pts) if pts else np.empty((0, 3))


stj = collect(STILT_JOBS)
sd = collect(TWIN_JOBS)
print(f"stilt points {len(stj)}, twin points {len(sd)}")

fig, (axH, axP) = plt.subplots(1, 2, figsize=(7.2, 3.0))

for ax, col in [(axH, 1), (axP, 2)]:
    if len(stj):
        ax.plot(stj[:, 0], stj[:, col], ".", color=STJ, ms=3, alpha=0.5)
        # smoothed trend (rolling median in step bins)
        for arr, c, lab in [(stj, STJ, "Stieltjes"), (sd, SDPA, "softmax twin")]:
            if not len(arr):
                continue
            b = np.linspace(0, arr[:, 0].max(), 12)
            idx = np.digitize(arr[:, 0], b)
            xs = [arr[idx == k, 0].mean() for k in range(1, len(b))
                  if (idx == k).any()]
            ys = [np.median(arr[idx == k, col]) for k in range(1, len(b))
                  if (idx == k).any()]
            ax.plot(xs, ys, "-", color=c, lw=2, label=lab)
    ax.set_xlabel("GRPO step")

axH.set_ylabel("attention entropy (mid-layer)")
axH.set_title("A  Attention entropy during RL\n(lower = sharper)",
              fontsize=9, loc="left")
axH.legend(frameon=False, fontsize=8)
axP.set_ylabel("mean max attention prob (mid-layer)")
axP.set_title("B  Attention peakiness during RL\n(higher = sharper)",
              fontsize=9, loc="left")

# quantitative summary
if len(stj) and len(sd):
    def enddrop(a, c):
        early = a[a[:, 0] <= a[:, 0].max() * 0.2][:, c].mean()
        late = a[a[:, 0] >= a[:, 0].max() * 0.8][:, c].mean()
        return early, late
    se, sl = enddrop(stj, 1)
    te, tl = enddrop(sd, 1)
    print(f"attn entropy early->late: stilt {se:.2f}->{sl:.2f} "
          f"(Δ{sl-se:+.2f}), twin {te:.2f}->{tl:.2f} (Δ{tl-te:+.2f})")

fig.tight_layout()
out = "/users/PAS2402/alexg/softmax/thesis/figures/fig_rl_attn_mechanism.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("wrote", out)
