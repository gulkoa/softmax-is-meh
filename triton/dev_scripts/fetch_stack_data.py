"""
Fetch a code-LM corpus to results/nl_data/stack_python.txt (~TARGET_MB of
Python source, one file after another, separated by NUL-free boundaries).

Primary: bigcode/the-stack-smol (data/python subset). If gated/unavailable,
fallback: codeparrot/codeparrot-clean-valid (public Python corpus).
Run on the login node (compute nodes have no internet).
"""
import os
import sys

OUT = "/users/PAS2402/alexg/softmax/softmax-is-meh/results/nl_data/stack_python.txt"
TARGET_MB = int(os.environ.get("TARGET_MB", "100"))
SEP = "\n\n# ---8<--- file boundary ---8<---\n\n"


def stream_to_file(ds_iter, field):
    total = 0
    target = TARGET_MB * 1_000_000
    with open(OUT, "w", encoding="utf-8", errors="ignore") as f:
        for ex in ds_iter:
            text = ex.get(field) or ""
            if not text:
                continue
            f.write(text)
            f.write(SEP)
            total += len(text) + len(SEP)
            if total >= target:
                break
    return total


def main():
    from datasets import load_dataset
    try:
        print("trying bigcode/the-stack-smol (data/python, streaming)...")
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python",
                          split="train", streaming=True)
        total = stream_to_file(ds, "content")
        print(f"the-stack-smol: wrote {total/1e6:.1f} MB -> {OUT}")
        return
    except Exception as e:
        print(f"the-stack-smol failed: {type(e).__name__}: {e}", file=sys.stderr)

    print("falling back to codeparrot/codeparrot-clean-valid (public python)...")
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                      streaming=True)
    total = stream_to_file(ds, "content")
    print(f"codeparrot fallback: wrote {total/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
