#!/usr/bin/env python3
"""Read the learned per-head score scales from train_gpt2_stieltjes checkpoints.

The trainer's learnable scale is eff = 1 + C*tanh(m/C) per head (C = the run's
--scale-cap, 15 by default; C = 0 means eff = 1 + m), multiplied into q before
the attention mapping (both mappings then apply 1/sqrt(head_dim)). This reads
only the scale_mult tensors (checkpoints are memory-mapped, so the 1.5 GB
optimizer state is never loaded) and reports eff per layer and head.

    python scripts/learned_scales.py LABEL=CKPT [LABEL=CKPT ...] [--json OUT]

Do not point it at a checkpoint that a running job may be rewriting: the
trainer saves in place, so a read can race a write.
"""
import argparse
import json
import math
import re

import torch


def effective_scales(path):
    blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    args = blob.get("args", {}) or {}
    cap = float(args.get("scale_cap", 15.0))
    layers = {}
    for key, m in blob["model"].items():
        if not key.endswith("scale_mult"):
            continue
        mm = re.search(r"(?:^|\.)(?:h|blocks|layers)\.(\d+)\.", key)
        layer = int(mm.group(1)) if mm else len(layers)
        m = m.float()
        eff = 1.0 + cap * torch.tanh(m / cap) if cap > 0 else 1.0 + m
        layers[layer] = [round(float(v), 4) for v in eff]
    return {"step": int(blob.get("step", -1)), "attn": args.get("attn"),
            "seed": args.get("seed"), "lr": args.get("lr"), "cap": cap,
            "layers": dict(sorted(layers.items()))}


def summarize(label, r):
    allv = [v for hs in r["layers"].values() for v in hs]
    if not allv:
        print(f"{label}: no learnable scale in this checkpoint")
        return
    n = len(allv)
    print(f"{label}: attn {r['attn']} seed {r['seed']} step {r['step']} "
          f"cap {r['cap']:g} | {len(r['layers'])} layers x "
          f"{n // max(len(r['layers']), 1)} heads")
    print(f"  all heads: mean {sum(allv) / n:.2f}  min {min(allv):.2f}  "
          f"max {max(allv):.2f}  |eff|<1: {sum(abs(v) < 1 for v in allv)}/{n}"
          f"  eff<0: {sum(v < 0 for v in allv)}/{n}")
    print("  layer   mean    min    max")
    for layer, hs in r["layers"].items():
        print(f"  {layer:5d} {sum(hs) / len(hs):6.2f} {min(hs):6.2f} "
              f"{max(hs):6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="LABEL=PATH")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for spec in a.ckpts:
        label, _, path = spec.partition("=")
        if not path:
            label, path = spec, spec
        r = effective_scales(path)
        r["ckpt"] = path
        out[label] = r
        summarize(label, r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
