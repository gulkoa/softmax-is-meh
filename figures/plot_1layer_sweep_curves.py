"""Plot val_accuracy and train_loss curves for all 1-layer max sweeps on Ascend."""
import os
import glob
import csv as _csv
import matplotlib.pyplot as plt

def read_metrics(path):
    eps, tls, vls, vas = [], [], [], []
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
    return eps, tls, vls, vas

RESULTS = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"
OUT_DIR = "/users/PAS2402/alexg/softmax/thesis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

dirs = sorted(glob.glob(f"{RESULTS}/max_1layer_*_nope_ascend"))

def label_of(d):
    n = os.path.basename(d).replace("max_1layer_", "").replace("_nope_ascend", "")
    return n

def seq_of(d):
    # expects ..._seq{N}_nope_ascend
    for part in os.path.basename(d).split("_"):
        if part.startswith("seq"):
            return int(part[3:])
    return 0

seqs = sorted({seq_of(d) for d in dirs})
fig, axes = plt.subplots(len(seqs), 1, figsize=(10, 2.5*len(seqs)), sharex=False)
if len(seqs) == 1: axes = [axes]

for ax, s in zip(axes, seqs):
    for d in dirs:
        if seq_of(d) != s: continue
        csv = os.path.join(d, "metrics.csv")
        if not os.path.isfile(csv): continue
        eps, tls, vls, vas = read_metrics(csv)
        if not eps: continue
        lbl = label_of(d).replace(f"_seq{s}", "")
        ls = "-" if "softmax" in lbl else "--"
        lw = 2.0 if "softmax" in lbl else 1.2
        ax.plot(eps, vas, label=lbl, linestyle=ls, linewidth=lw, alpha=0.85)
    ax.set_title(f"train seq={s}")
    ax.set_ylabel("val_accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

axes[-1].set_xlabel("epoch")
fig.suptitle("1-layer max — Ascend 1L-scaling + q-sweep training curves")
out = os.path.join(OUT_DIR, "1layer_sweep_curves_ascend.png")
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"wrote {out}")
