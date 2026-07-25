"""Held-out reasoning eval: GSM8K TEST split + fresh-seed logic tasks
(unseen by construction) + ARC-Easy TEST slice. Reports acc@1 (mean
over k samples) and solve@k per channel. Reuses the GRPO harness.

Usage: python eval_reason_stilt.py <ckpt.pt> [--k 8] [--n 300]
"""
import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT, FW_DIR  # noqa: E402
from grpo_code_stilt import sample_batch  # noqa: E402
from grpo_reason_stilt import (  # noqa: E402
    build_prompt, extract_answer, is_correct)
import gen_logic_tasks as glt  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEVICE = torch.device("cuda")


def held_out_tasks(n):
    tasks = []
    rng = random.Random(999)                    # unseen seed
    for i in range(n // 3):
        t = glt.make_task(rng, i, rng.choice([1, 2, 3]))
        t["channel"] = "logic"
        t["question"] = t["question"]
        tasks.append(t)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for i, ex in enumerate(ds.select(range(n // 3))):
        ans = ex["answer"].split("####")[-1].strip()
        tasks.append({"task_id": f"gsm8kT-{i}", "channel": "gsm8k",
                      "question": ex["question"], "answer": ans,
                      "aliases": [ans]})
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    for i, ex in enumerate(ds.select(range(n - 2 * (n // 3)))):
        labs, txts = ex["choices"]["label"], ex["choices"]["text"]
        key = ex["answerKey"]
        txt = txts[labs.index(key)] if key in labs else ""
        tasks.append({"task_id": f"arcT-{i}", "channel": "arc",
                      "question": ex["question"],
                      "choices": list(zip(labs, txts)),
                      "answer": key, "aliases": [key, txt]})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=300)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("gpt2")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    model.eval()

    tasks = held_out_tasks(args.n)
    stats = {}
    for ti, task in enumerate(tasks):
        outs, _, _ = sample_batch(model, tok, [build_prompt(task)],
                                  args.k, args.max_new, cfg.ctx)
        hits = [is_correct(extract_answer(o), task) for o in outs]
        s = stats.setdefault(task["channel"], [0.0, 0, 0])
        s[0] += float(np.mean(hits))            # acc@1 contribution
        s[1] += 1 if any(hits) else 0           # solve@k
        s[2] += 1
        if ti % 50 == 0:
            print(f"{ti}/{len(tasks)}", {c: (round(v[0] / max(v[2], 1), 3),
                                             f"{v[1]}/{v[2]}")
                                         for c, v in stats.items()},
                  flush=True)
    agg = {c: {"acc1": v[0] / v[2], "solve_at_k": v[1] / v[2],
               "n": v[2]} for c, v in stats.items()}
    out = args.ckpt.replace(".pt", "_reasoneval.json")
    json.dump(agg, open(out, "w"))
    print("SUMMARY", json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
