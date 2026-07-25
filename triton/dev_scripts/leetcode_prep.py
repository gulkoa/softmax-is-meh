"""LeetCodeDataset -> MBPP-schema curriculum for code-RL round 3
(user pick over KodCode). Temporal split: TRAIN = pre-2024 problems,
HELDOUT = 2024+ (contamination-safe eval by construction).

Adaptation: solutions are `class Solution` style; we bind
`candidate = <entry_point>` in setup so the check-body asserts run as
independent lines (dense per-test reward fractions preserved). Every
row is VALIDATED by running its own reference solution against the
extracted asserts in the sandbox — extraction bugs self-filter.

Row schema: {task_id, text, starter_code, test_list,
test_setup_code, code, difficulty, entry_point}.

Usage: python leetcode_prep.py [--difficulty Easy,Medium]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_exec_sandbox import run_tests  # noqa: E402

os.environ.setdefault(
    "HF_HOME", "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/hf_cache")
from datasets import load_dataset  # noqa: E402

IMPORTS = ("from typing import *\nfrom collections import *\n"
           "from itertools import *\nfrom functools import *\n"
           "import math\nimport string\nimport re\n"
           "import heapq\nimport bisect\n")


def extract_asserts(test_src):
    lines = []
    for ln in test_src.splitlines():
        st = ln.strip()
        if st.startswith("assert candidate"):
            lines.append(st.replace("assert candidate",
                                    "assert candidate"))
        elif st.startswith(("def check", "#")) or not st:
            continue
        else:
            return None            # non-assert logic inside check
    return lines if len(lines) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="Easy,Medium")
    ap.add_argument("--out-train", default="leetcode_rl_train.json")
    ap.add_argument("--out-heldout", default="leetcode_rl_heldout.json")
    args = ap.parse_args()
    diffs = set(args.difficulty.split(","))

    ds = load_dataset("newfacade/LeetCodeDataset", split="train")
    train, heldout = [], []
    flat_fail = val_fail = 0
    for ex in ds:
        if ex["difficulty"] not in diffs:
            continue
        asserts = extract_asserts(ex["test"] or "")
        if not asserts:
            flat_fail += 1
            continue
        setup = IMPORTS
        binding = f"\ncandidate = {ex['entry_point']}\n"
        ref = setup + ex["completion"] + binding
        frac, ok = run_tests(ref, asserts)
        if frac != 1.0:
            val_fail += 1
            continue
        row = {"task_id": f"lc-{ex['task_id']}",
               "text": (ex["problem_description"].strip()
                        + "\n\nComplete this starter code:\n"
                        + ex["starter_code"].strip()),
               "starter_code": ex["starter_code"],
               "test_list": asserts,
               "test_setup_code": setup,
               "post_code": binding,
               "code": ex["completion"],
               "entry_point": ex["entry_point"],
               "difficulty": ex["difficulty"]}
        year = int(str(ex["estimated_date"])[:4])
        (train if year < 2024 else heldout).append(row)
    json.dump(train, open(args.out_train, "w"), indent=1)
    json.dump(heldout, open(args.out_heldout, "w"), indent=1)
    dmix = {}
    for t in train:
        dmix[t["difficulty"]] = dmix.get(t["difficulty"], 0) + 1
    print(f"DONE: train {len(train)} {dmix} | heldout {len(heldout)} | "
          f"flat-fail {flat_fail} val-fail {val_fail}", flush=True)


if __name__ == "__main__":
    main()
