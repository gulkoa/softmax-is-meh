"""Reasoning-format SFT data for Stilt code-RL: augment MBPP-train and
synthetic tasks with rule-based <think> traces (function name + arity
from the tests, approach outline from the reference solution's
constructs). The traces are mechanical — they teach the FORMAT and
grounded pre-code analysis; whether think-space helps is then GRPO's
question, not an assumption.

Usage: python gen_reasoning_traces.py [--synthetic tasks.json]
           [--out reasoning_code_sft.json]
Output rows: {text, test_list, code, think} — sft_code_stilt
--reasoning consumes `think`.
"""
import argparse
import json
import os
import re

os.environ.setdefault(
    "HF_HOME", "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/hf_cache")
from datasets import load_dataset  # noqa: E402

CONSTRUCTS = [
    (r"\bre\.", "use the re module for pattern matching"),
    (r"\bdef \w+\(.*\).*\n.*\bdef ", "define a helper function"),
    (r"\[.+ for .+ in .+\]", "build the result with a comprehension"),
    (r"\bfor .+ in .+:", "iterate with a for loop"),
    (r"\bwhile .+:", "use a while loop"),
    (r"\bsorted?\(", "sort the data"),
    (r"\b(dict|\{\})", "track values with a dict"),
    (r"\bset\(", "use a set for uniqueness"),
    (r"\bmax\(|\bmin\(", "take a max/min"),
    (r"\bsum\(", "accumulate with sum"),
    (r"%\s*\d|\bdivmod\b", "use modular arithmetic"),
    (r"\[::-1\]|reversed\(", "reverse the sequence"),
    (r"\.join\(", "join parts into a string"),
    (r"\.split\(", "split the string"),
    (r"\breturn .+ if .+ else ", "return via a conditional expression"),
    (r"\blambda\b", "use a small lambda"),
]


def fname_and_arity(test_list):
    for t in test_list:
        m = re.search(r"assert\s+(?:\w+\s*\(\s*)?(\w+)\s*\((.*)", t)
        if m and m.group(1) not in ("len", "set", "tuple", "list", "abs",
                                    "math", "round", "str", "int"):
            depth, args, cur = 0, 0, ""
            for ch in m.group(2):
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    if depth == 0:
                        break
                    depth -= 1
                elif ch == "," and depth == 0:
                    args += 1
                cur += ch
            return m.group(1), (args + 1 if cur.strip() else 0)
    return None, None


def make_think(problem):
    text = problem["text"].strip()
    fname, arity = fname_and_arity(problem["test_list"])
    parts = [f"Task: {text}"]
    if fname:
        parts.append(f"The tests call {fname}() with "
                     f"{arity} argument(s), e.g. "
                     f"{problem['test_list'][0].strip()}")
    plans = [desc for pat, desc in CONSTRUCTS
             if re.search(pat, problem["code"])]
    if plans:
        parts.append("Plan: " + "; ".join(plans[:3]) + ".")
    else:
        parts.append("Plan: a direct one-liner should work.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--out", default="reasoning_code_sft.json")
    args = ap.parse_args()

    ds = load_dataset("google-research-datasets/mbpp", "full")
    problems = list(ds["train"])
    if args.synthetic:
        problems += json.load(open(args.synthetic))
    rows = []
    for p in problems:
        rows.append({"text": p["text"], "test_list": p["test_list"],
                     "test_setup_code": p.get("test_setup_code", "") or "",
                     "code": p["code"], "think": make_think(p)})
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"wrote {len(rows)} reasoning rows -> {args.out}")
    print("SAMPLE THINK:", rows[0]["think"][:200])


if __name__ == "__main__":
    main()
