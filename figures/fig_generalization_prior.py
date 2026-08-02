"""Publication figure: heavy-tailed attention as a generalization prior.
The cross-cutting thesis theme — at MATCHED training, Stieltjes wins the
HELD-OUT comparison against its softmax twin in every regime.

Panel A (SFT, the sharpest datapoint): identical SFT ladders give
near-equal TRAIN-solvable sets (122 vs 121/374) but a large HELD-OUT
gap (MBPP-test pass@1 3.2 vs 1.88) — train competence equalized,
generalization did not.
Panel B (pretraining, 355M): Stieltjes's relative advantage over the
softmax twin on each held-out eval — all positive.

Login-node only. Numbers from committed findings (355m-headtohead,
grpo-v1-collapse side-by-side). Output: thesis/figures/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STJ = "#0072B2"
SDPA = "#D55E00"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))

# --- Panel A: matched train, divergent held-out ---
groups = ["train-solvable\n(/374)", "held-out MBPP\npass@1 (%)"]
stj_vals = [122, 3.2]
sdpa_vals = [121, 1.88]
# two independent y-scales via twin axes (counts vs %)
x = [0, 1]
w = 0.36
axA.bar([x[0] - w / 2], [stj_vals[0]], w, color=STJ)
axA.bar([x[0] + w / 2], [sdpa_vals[0]], w, color=SDPA)
axA.set_ylabel("train-solvable (count)", color="#444")
axA.set_ylim(0, 200)
axA2 = axA.twinx()
axA2.spines["top"].set_visible(False)
axA2.bar([x[1] - w / 2], [stj_vals[1]], w, color=STJ)
axA2.bar([x[1] + w / 2], [sdpa_vals[1]], w, color=SDPA)
axA2.set_ylabel("held-out pass@1 (%)", color="#444")
axA2.set_ylim(0, 5)
axA.set_xticks(x)
axA.set_xticklabels(groups)
axA.axvline(0.5, color="#DDD", lw=0.8)
axA.text(0, 122 + 6, "≈equal", ha="center", fontsize=8, color="#333")
axA2.annotate("+70%", (1 + w / 2, 3.2), (1.15, 3.6), color=STJ,
              fontsize=8, arrowprops=dict(arrowstyle="->", color=STJ,
                                          lw=0.8))
axA.set_title("A  Matched SFT: equal train competence,\n"
              "divergent generalization", fontsize=9, loc="left")
# shared legend
from matplotlib.patches import Patch
axA.legend(handles=[Patch(color=STJ, label="Stieltjes"),
                    Patch(color=SDPA, label="softmax twin")],
           frameon=False, fontsize=8, loc="upper right")

# --- Panel B: pretraining held-out advantage ---
metrics = ["FineWeb-val\nppl", "WikiText-103\nppl", "LAMBADA\nacc"]
# stilt relative advantage (%): ppl lower=better, acc higher=better
stj_p = [16.54, 26.09, 0.302]
sdpa_p = [17.05, 27.14, 0.289]
adv = [100 * (sdpa_p[0] - stj_p[0]) / sdpa_p[0],
       100 * (sdpa_p[1] - stj_p[1]) / sdpa_p[1],
       100 * (stj_p[2] - sdpa_p[2]) / sdpa_p[2]]
bars = axB.bar(range(len(metrics)), adv, color=STJ, width=0.6)
for i, a in enumerate(adv):
    axB.text(i, a + 0.1, f"+{a:.1f}%", ha="center", fontsize=8, color=STJ)
axB.axhline(0, color="#888", lw=0.8)
axB.set_xticks(range(len(metrics)))
axB.set_xticklabels(metrics)
axB.set_ylabel("Stieltjes advantage over\nsoftmax twin (%)")
axB.set_ylim(0, 6)
axB.set_title("B  Pretraining (355M): held-out wins\n"
              "(matched data, best stable lr)", fontsize=9, loc="left")

fig.tight_layout()
out = "/users/PAS2402/alexg/softmax/thesis/figures/fig_generalization_prior.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("wrote", out)
