"""Aggregate accuracy_all (full-sequence) across the embd-scan checkpoints.

The existing accuracy_eval_seq128_arr120_first1.json files already contain
accuracy_fixed, accuracy_all, and accuracy_input_echo from one eval pass —
we just need to read them.

Prints a wide table: n_embd x (softmax + stj q in {1,2,4,8,16,32}) showing
accuracy_all per seed, plus mean/std where seeds >= 2.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
PAT = re.compile(
    r"^max_1layer_"
    r"(?P<attn>softmax|stieltjes)"
    r"(?:_q(?P<q>[\d.]+))?"
    r"_seq128_embd(?P<embd>\d+)_h(?P<head>\d+)"
    r"(?:_seed(?P<seed>\d+))?"
    r"_initfix_ascend$"
)

def main() -> None:
    rows = defaultdict(lambda: defaultdict(list))
    for p in RESULTS.iterdir():
        if not p.is_dir():
            continue
        m = PAT.match(p.name)
        if not m:
            continue
        eval_json = p / "accuracy_eval_seq128_arr120_first1.json"
        if not eval_json.exists():
            continue
        try:
            data = json.loads(eval_json.read_text())
        except Exception:
            continue
        attn = m.group("attn")
        q = float(m.group("q")) if m.group("q") else None
        embd = int(m.group("embd"))
        seed = int(m.group("seed")) if m.group("seed") else 42
        label = "softmax" if attn == "softmax" else f"stj_q{q:g}"
        rows[embd][label].append({
            "seed": seed,
            "accuracy_fixed": data.get("accuracy_fixed"),
            "accuracy_all": data.get("accuracy_all"),
            "accuracy_input_echo": data.get("accuracy_input_echo"),
        })

    print(f"{'n_embd':>6} | {'config':<12} | {'seed':>4} | "
          f"{'acc_fixed':>10} | {'acc_all':>10} | {'acc_input':>10}")
    print("-" * 70)
    for embd in sorted(rows):
        for label in sorted(rows[embd]):
            entries = sorted(rows[embd][label], key=lambda e: e["seed"])
            for e in entries:
                print(f"{embd:>6} | {label:<12} | {e['seed']:>4} | "
                      f"{e['accuracy_fixed']:>10.4f} | "
                      f"{e['accuracy_all']:>10.4f} | "
                      f"{e['accuracy_input_echo']:>10.4f}")
            if len(entries) >= 2:
                m_all = mean(e["accuracy_all"] for e in entries)
                s_all = pstdev(e["accuracy_all"] for e in entries)
                m_fix = mean(e["accuracy_fixed"] for e in entries)
                s_fix = pstdev(e["accuracy_fixed"] for e in entries)
                print(f"{embd:>6} | {label:<12} | mean | "
                      f"{m_fix:>10.4f} | {m_all:>10.4f} | "
                      f"{'':>10}")
                print(f"{embd:>6} | {label:<12} |  std | "
                      f"{s_fix:>10.4f} | {s_all:>10.4f} | "
                      f"{'':>10}")
            print()

if __name__ == "__main__":
    main()
