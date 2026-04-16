"""Plot mean attention entropy vs eval_seq for Velickovic-replication sweep.

Reads entropy CSVs from
  results/subtle_needle_1layer_{softmax,stieltjes_q*}_seq128_nope_ascend/entropy_seq{S}/

Produces:
  thesis/figures/velickovic_entropy_vs_seq.png
  thesis/figures/velickovic_output_std_vs_seq.png (if available)

Each curve = one (attn, q) config; x = eval_seq, y = mean entropy.
"""
from __future__ import annotations

import csv
import glob
import os
import re

import matplotlib.pyplot as plt

RESULTS = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"
OUT_DIR = "/users/PAS2402/alexg/softmax/thesis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    ("softmax", None, "black", "-"),
    ("stieltjes", 4.0, "tab:blue", "--"),
    ("stieltjes", 8.0, "tab:green", "--"),
    ("stieltjes", 16.0, "tab:orange", "--"),
    ("stieltjes", 24.0, "tab:red", "--"),
    ("stieltjes", 32.0, "tab:purple", "--"),
]

EVAL_SEQS = [128, 512, 2048]  # 8192 OOMs analyze.py (O(N²) storage)


def model_dir(attn, q):
    if attn == "softmax":
        return f"{RESULTS}/subtle_needle_1layer_softmax_seq128_nope_ascend"
    return f"{RESULTS}/subtle_needle_1layer_stieltjes_q{q}_seq128_nope_ascend"


def load_mean_entropy(model_dir, eval_seq):
    """Return (mean_entropy, None) if available, else (None, None)."""
    d = f"{model_dir}/entropy_seq{eval_seq}"
    # analyze.py writes entropy.csv with columns layer,head,entropy
    csv_path = f"{d}/entropy.csv"
    if not os.path.isfile(csv_path):
        # Try alternative filenames
        for f in glob.glob(f"{d}/*.csv"):
            if "entropy" in f or "stat" in f:
                csv_path = f
                break
        else:
            return None
    try:
        with open(csv_path) as f:
            r = csv.DictReader(f)
            rows = list(r)
    except Exception:
        return None
    if not rows:
        return None
    # Find an entropy column
    e_key = None
    for k in rows[0]:
        if "entropy" in k.lower():
            e_key = k
            break
    if e_key is None:
        return None
    vals = [float(row[e_key]) for row in rows if row.get(e_key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


fig, ax = plt.subplots(figsize=(8, 5))

missing = []
for attn, q, color, ls in CONFIGS:
    d = model_dir(attn, q)
    if not os.path.isdir(d):
        missing.append(f"{attn}_{q}: no model dir")
        continue
    ys, xs = [], []
    for s in EVAL_SEQS:
        e = load_mean_entropy(d, s)
        if e is not None:
            xs.append(s)
            ys.append(e)
    if not xs:
        missing.append(f"{attn}_{q}: no entropy data")
        continue
    label = "softmax" if attn == "softmax" else f"stj q={q:g}"
    ax.plot(xs, ys, marker="o", color=color, linestyle=ls, label=label, linewidth=1.5)

ax.set_xscale("log", base=2)
ax.set_xlabel("eval sequence length (log scale)")
ax.set_ylabel("mean attention entropy (nats)")
ax.set_title("1-layer subtle-needle (trained at seq=128): attention entropy vs OOD eval seq")
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=9)

# Overlay log(N) reference — the max possible entropy at size N
import numpy as np
Ns = np.array(EVAL_SEQS, dtype=float)
ax.plot(Ns, np.log(Ns), color="gray", linestyle=":", alpha=0.5, label="log(N) = max entropy")
ax.legend(loc="best", fontsize=9)

out = os.path.join(OUT_DIR, "velickovic_entropy_vs_seq.png")
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"wrote {out}")

if missing:
    print("\nMissing data for:")
    for m in missing:
        print(f"  - {m}")
