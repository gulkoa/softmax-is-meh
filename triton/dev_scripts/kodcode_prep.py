"""KodCode -> MBPP-schema curriculum for code-RL round 3 (fused
execution+judge reward, robust dataset).

Streams KodCode-V1, keeps easy/medium rows, flattens pytest-style
tests into assert lines, and VALIDATES each row by running its own
reference solution against the extracted tests in the sandbox — only
rows whose reference passes 100% survive (test extraction is thus
self-validating). Output rows: {task_id, text, test_list,
test_setup_code, code, difficulty}.

Usage: python kodcode_prep.py [--n-keep 1500] [--scan 20000]
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


def extract_asserts(test_src):
    """Flatten pytest functions into standalone assert lines; None if
    the tests use anything we can't flatten (fixtures, loops, ...)."""
    if re.search(r"@pytest|fixture|parametrize|raises|import (?!solution)",
                 test_src):
        return None
    lines = []
    for ln in test_src.splitlines():
        st = ln.strip()
        if st.startswith("assert "):
            lines.append(st)
        elif st and not (st.startswith(("def test", "from solution",
                                        "import solution", "#"))
                         or st == ""):
            # any other statement inside tests (setup vars etc.) —
            # too complex to flatten safely
            if not st.startswith("assert"):
                return None
    return lines if len(lines) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-keep", type=int, default=1500)
    ap.add_argument("--scan", type=int, default=20000)
    ap.add_argument("--out", default="kodcode_rl_v1.json")
    args = ap.parse_args()

    ds = load_dataset("KodCode/KodCode-V1", split="train", streaming=True)
    kept, scanned, flat_fail, val_fail = [], 0, 0, 0
    for ex in ds:
        scanned += 1
        if scanned > args.scan or len(kept) >= args.n_keep:
            break
        if ex.get("gpt_difficulty") not in ("easy", "medium"):
            continue
        asserts = extract_asserts(ex["test"] or "")
        if not asserts:
            flat_fail += 1
            continue
        frac, ok = run_tests(ex["solution"], asserts)
        if frac != 1.0:
            val_fail += 1
            continue
        kept.append({"task_id": f"kod-{ex['question_id']}",
                     "text": ex["question"].strip(),
                     "test_list": asserts, "test_setup_code": "",
                     "code": ex["solution"],
                     "difficulty": ex["gpt_difficulty"]})
        if len(kept) % 200 == 0:
            print(f"kept {len(kept)} (scanned {scanned}, "
                  f"flat-fail {flat_fail}, val-fail {val_fail})",
                  flush=True)
    json.dump(kept, open(args.out, "w"), indent=1)
    mix = {}
    for t in kept:
        mix[t["difficulty"]] = mix.get(t["difficulty"], 0) + 1
    print(f"DONE: {len(kept)} validated tasks -> {args.out} | mix {mix} "
          f"| scanned {scanned}, flat-fail {flat_fail}, "
          f"val-fail {val_fail}", flush=True)


if __name__ == "__main__":
    main()
