#!/usr/bin/env python
"""Quick aggregator for needle accuracy_fixed.json files. Run from anywhere."""
import json
import re
from pathlib import Path

ROOT = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")

rows = []
for j in sorted(ROOT.glob("needle_*/accuracy_fixed.json")):
    d = json.loads(j.read_text())
    name = j.parent.name
    seq = re.search(r"seq(\d+)", name)
    seq = int(seq.group(1)) if seq else None
    nope = "_nope_" in name or name.endswith("_nope_ascend")
    rows.append((seq or 0, name, d.get("accuracy_fixed"), nope))

print(f"{'seq':>5s}  {'run':<55s} {'fixed':>8s}  pe")
for seq, name, acc, nope in sorted(rows):
    pe = "NoPE" if nope else "learned"
    print(f"  {seq:>3d}  {name:<55s}  {acc:.4f}  {pe}")
