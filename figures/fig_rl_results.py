"""Publication figure: 355M RL results (all held-out, shipped models).
Panel A — code-RL MBPP-test pass@1 vs GRPO step: stilt climbs, softmax
twin regresses below its own baseline (mapping divergence under
identical treatment). Panel B — reasoning-RL held-out gains
(GSM8K-test / ARC-Easy-test / fresh logic), before vs after.

Login-node only (no GPU). Numbers from the committed findings:
findings/2026-07-24-grpo-v1-collapse, 2026-07-25-reasoning-rl-flagship.
Output: thesis/figures/fig_rl_results.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe
STJ = "#0072B2"      # blue
SDPA = "#D55E00"     # vermillion
GAIN = "#009E73"     # green

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})

steps = [0, 100, 200, 300, 400]
stj = [3.2, 3.6, 3.7, 4.07, 3.72]      # MBPP-test pass@1 (%)
sdpa = [1.88, 2.57, 1.77, 1.70, 1.80]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))

# --- Panel A: code-RL divergence ---
axA.axhline(stj[0], color=STJ, ls=":", lw=1, alpha=0.6)
axA.axhline(sdpa[0], color=SDPA, ls=":", lw=1, alpha=0.6)
axA.plot(steps, stj, "-o", color=STJ, ms=5, lw=2, label="Stieltjes (q=4)")
axA.plot(steps, sdpa, "-s", color=SDPA, ms=5, lw=2, label="softmax twin")
axA.annotate("selected\n+27%", (300, 4.07), (250, 4.55),
             color=STJ, fontsize=8, ha="center",
             arrowprops=dict(arrowstyle="->", color=STJ, lw=0.8))
axA.annotate("regresses below\nown baseline", (300, 1.70), (110, 1.15),
             color=SDPA, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=SDPA, lw=0.8))
axA.set_xlabel("GRPO step")
axA.set_ylabel("MBPP-test pass@1 (%)")
axA.set_title("A  Code-RL: identical treatment,\nopposite outcomes",
              fontsize=9, loc="left")
axA.set_xticks(steps)
axA.set_ylim(0.8, 5.0)
axA.legend(frameon=False, fontsize=8, loc="center right")

# --- Panel B: reasoning-RL gains ---
tasks = ["GSM8K\n(test)", "ARC-Easy\n(test)", "logic\n(fresh)"]
before = [3.0, 32.5, 5.5]
after = [6.4, 58.9, 10.0]
x = range(len(tasks))
w = 0.38
axB.bar([i - w / 2 for i in x], before, w, color="#BBBBBB",
        label="before RL")
axB.bar([i + w / 2 for i in x], after, w, color=GAIN, label="after RL")
for i, (b, a) in enumerate(zip(before, after)):
    axB.text(i + w / 2, a + 1.2, f"+{100*(a-b)/b:.0f}%", ha="center",
             fontsize=7.5, color=GAIN)
axB.set_xticks(list(x))
axB.set_xticklabels(tasks)
axB.set_ylabel("held-out acc@1 (%)")
axB.set_title("B  Reasoning-RL: fused verify+judge\nreward (stj-355m)",
              fontsize=9, loc="left")
axB.set_ylim(0, 68)
axB.legend(frameon=False, fontsize=8, loc="upper left")

fig.tight_layout()
out = "/users/PAS2402/alexg/softmax/thesis/figures/fig_rl_results.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("wrote", out)
