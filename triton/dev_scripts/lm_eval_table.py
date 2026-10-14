"""Aggregate lm-evaluation-harness results (results/lm_eval/<tag>/**/results_*.json)
into one markdown table: one row per model, one column per task (acc, or
acc_norm where the task defines it, bits_per_byte for wikitext). Login node.
  python lm_eval_table.py [--root results/lm_eval] [--norm]  (--norm: prefer acc_norm)
"""
import argparse, glob, json, os

TASKS = [("hellaswag", "HellaSwag"), ("piqa", "PIQA"), ("arc_easy", "ARC-E"), ("arc_challenge", "ARC-C"),
         ("sciq", "SciQ"), ("openbookqa", "OBQA"), ("winogrande", "WinoG"), ("boolq", "BoolQ"),
         ("lambada_openai", "LAMBADA"), ("commonsense_qa", "CSQA"), ("mmlu", "MMLU"), ("wikitext", "WT2 BPB")]
ORDER = ["gpt2", "stilt.1-124m", "gpt2-medium", "stilt.1-355m", "stilt.1-355m-softmax-baseline",
         "stilt.1.1-355m-nope", "stilt.1.1-355m-nope-softmax-baseline", "stilt.1.1-355m-rope",
         "stilt.1.1-355m-nope-qk", "stilt.1.1-355m-nope-qk-softmax-baseline"]


def load(root):
    out = {}
    for tagdir in sorted(glob.glob(f"{root}/*/")):
        tag = os.path.basename(tagdir.rstrip("/"))
        if tag.startswith("_"):
            continue
        files = sorted(glob.glob(f"{tagdir}/**/results_*.json", recursive=True))
        if not files:
            continue
        r = json.load(open(files[-1]))["results"]
        out[tag] = r
    return out


def cell(r, task, norm):
    if task not in r:
        return "—"
    d = r[task]
    if task == "wikitext":
        return f"{d['bits_per_byte,none']:.3f}"
    key = "acc_norm,none" if norm and "acc_norm,none" in d else "acc,none"
    v = d[key]
    return f"{100 * v:.1f}" + ("ⁿ" if key.startswith("acc_norm") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm_eval")
    ap.add_argument("--norm", action="store_true")
    a = ap.parse_args()
    res = load(a.root)
    tags = [t for t in ORDER if t in res] + sorted(t for t in res if t not in ORDER)
    print("| model | " + " | ".join(n for _, n in TASKS) + " | mean acc (9 tasks) |")
    print("|---|" + "---|" * (len(TASKS) + 1))
    for t in tags:
        r = res[t]
        accs = [r[k]["acc,none"] for k, _ in TASKS if k in r and k not in ("wikitext", "mmlu", "commonsense_qa")]
        mean = f"{100 * sum(accs) / len(accs):.1f}" if accs else "—"
        print(f"| {t} | " + " | ".join(cell(r, k, a.norm) for k, _ in TASKS) + f" | {mean} |")
    print(f"\n{len(tags)} models; accuracies in %, zero-shot; ⁿ = acc_norm; mean over HellaSwag, PIQA, ARC-E, ARC-C, SciQ, OBQA, WinoGrande, BoolQ, LAMBADA (acc).")


if __name__ == "__main__":
    main()
