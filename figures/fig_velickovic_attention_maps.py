"""Velickovic-style attention-map grid: rows = attn configs, cols = eval_seq.

For each cell, we show the attention weights from the LAST query position over
key positions, aggregated as (num_samples, n_key_positions) heatmap. Similar to
simple.ipynb's activation plot but spanning our stj-q sweep.

Expects pre-extracted patterns at
  results/subtle_needle_1layer_<tag>_seq128_nope_ascend/attn_patterns/attn_last_query_seq{S}.pt
Produces
  thesis/figures/velickovic_attention_map_grid.png
"""
from __future__ import annotations

import glob
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"
OUT_DIR = "/users/PAS2402/alexg/softmax/thesis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    ("softmax", None, "softmax"),
    ("stieltjes", 4.0, "stj q=4"),
    ("stieltjes", 8.0, "stj q=8"),
    ("stieltjes", 16.0, "stj q=16"),
    ("stieltjes", 24.0, "stj q=24"),
    ("stieltjes", 32.0, "stj q=32"),
]
EVAL_SEQS = [128, 512, 2048, 8192]
KEEP_KEYS = 32  # we'll show the top-KEEP_KEYS attended keys (like simple.ipynb top-16)


def model_dir(attn, q):
    if attn == "softmax":
        return f"{RESULTS}/subtle_needle_1layer_softmax_seq128_nope_ascend"
    return f"{RESULTS}/subtle_needle_1layer_stieltjes_q{q}_seq128_nope_ascend"


def load_attn_for_cell(attn, q, seq_len):
    """Return (num_samples, KEEP_KEYS) array of attention weights over top-k keys,
    aggregated across all layers+heads (mean). None if missing."""
    d = model_dir(attn, q)
    pt = os.path.join(d, "attn_patterns", f"attn_last_query_seq{seq_len}.pt")
    if not os.path.isfile(pt):
        return None
    try:
        obj = torch.load(pt, map_location="cpu")
    except Exception:
        return None
    w = obj["attn_last_query"]  # (L, B, H, T)
    # Mean over layers and heads → (B, T)
    w = w.mean(dim=(0, 2))
    # For each sample, select top-KEEP_KEYS attended positions to visualize
    # density structure (like simple.ipynb used sorted top-16)
    B, T = w.shape
    k = min(KEEP_KEYS, T)
    # Sort each row descending; keep top-k
    sorted_w, _ = torch.sort(w, dim=-1, descending=True)
    top_k = sorted_w[:, :k]
    return top_k.numpy()


def main():
    fig, axes = plt.subplots(
        len(CONFIGS), len(EVAL_SEQS),
        figsize=(3.0 * len(EVAL_SEQS), 1.8 * len(CONFIGS)),
        squeeze=False,
    )

    # Global vmax for consistent colorscale within each row? Or global?
    # Use per-row vmax so row comparisons across seqs are meaningful.
    for ri, (attn, q, label) in enumerate(CONFIGS):
        # Collect this row's data
        row_data = [load_attn_for_cell(attn, q, s) for s in EVAL_SEQS]
        present = [x for x in row_data if x is not None]
        if present:
            row_vmax = max(x.max() for x in present)
        else:
            row_vmax = 1.0
        for ci, s in enumerate(EVAL_SEQS):
            ax = axes[ri, ci]
            arr = row_data[ci]
            if arr is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=16, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.imshow(arr, cmap="Blues", aspect="auto",
                          vmin=0, vmax=row_vmax)
                ax.set_xticks([])
                ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"eval_seq={s}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle(
        "Attention weight patterns — last query over top-32 keys "
        "(trained at seq=128, 1-layer subtle-needle)",
        fontsize=12,
    )
    out = os.path.join(OUT_DIR, "velickovic_attention_map_grid.pdf")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
