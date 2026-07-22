"""Stage-3 code evaluator for the RL line (plan: thesis/findings/
2026-07-20-plan-rl-coding-stilt11.md): MBPP-test and HumanEval pass@k
(unbiased estimator, Chen et al. 2021) for any Stilt/twin checkpoint.
Reuses the GRPO harness's sampling/extraction and the exec sandbox.

Usage:
  python eval_code_stilt.py <ckpt.pt> [--dataset mbpp|humaneval|both]
      [--n 20] [--k 1,10] [--temperature 0.8] [--limit N]

Writes <ckpt>_codeeval_<dataset>.json with per-problem pass counts.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from types import SimpleNamespace  # noqa: E402

from train_gpt2_stieltjes import GPT, FW_DIR  # noqa: E402
from grpo_code_stilt import (  # noqa: E402
    U, A, build_prompt, extract_code, sample_batch)
from code_exec_sandbox import run_tests  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEVICE = torch.device("cuda")


def pass_at_k(n, c, k):
    """Unbiased pass@k: 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def he_prompt(problem):
    return (U + "Complete this Python function. Reply with the full "
            "function in a code block.\n```python\n"
            + problem["prompt"].rstrip() + "\n```\n" + A)


def he_passes(code, problem):
    # sandbox tests are line-wise; fold the multi-line HumanEval check
    # harness into the candidate program and use a sentinel test
    prog = (code + "\n\n" + problem["test"]
            + f"\n\ncheck({problem['entry_point']})\n")
    frac, _ = run_tests(prog, ["assert True"])
    return frac == 1.0


def mbpp_passes(code, problem):
    frac, _ = run_tests(code, problem["test_list"],
                        problem.get("test_setup_code", "") or "")
    return frac == 1.0


def evaluate(model, tok, problems, mk_prompt, passes, n, ks, max_new, ctx,
             temperature, tag):
    results = {}
    for pi, prob in enumerate(problems):
        outs, _, _ = sample_batch(model, tok, [mk_prompt(prob)], n,
                                  max_new, ctx, temperature=temperature)
        c = sum(passes(extract_code(o), prob) for o in outs)
        pid = prob.get("task_id", pi)
        results[str(pid)] = {"n": n, "c": int(c)}
        if pi % 20 == 0:
            agg = {k: float(np.mean([pass_at_k(r["n"], r["c"], k)
                                     for r in results.values()]))
                   for k in ks}
            print(f"[{tag}] {pi}/{len(problems)} {agg}", flush=True)
    agg = {f"pass@{k}": float(np.mean([pass_at_k(r["n"], r["c"], k)
                                       for r in results.values()]))
           for k in ks}
    return results, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--dataset", default="both",
                    choices=["mbpp", "humaneval", "both"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", default="1,10")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    assert args.n >= max(ks), "need n >= max k for the estimator"

    tok = AutoTokenizer.from_pretrained("gpt2")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    model.eval()

    summary = {}
    if args.dataset in ("mbpp", "both"):
        ds = load_dataset("google-research-datasets/mbpp", "full")
        probs = list(ds["test"])[: args.limit]
        res, agg = evaluate(model, tok, probs, build_prompt, mbpp_passes,
                            args.n, ks, args.max_new, cfg.ctx,
                            args.temperature, "mbpp")
        out = args.ckpt.replace(".pt", "_codeeval_mbpp.json")
        json.dump({"agg": agg, "per_problem": res}, open(out, "w"))
        print(f"MBPP-test ({len(probs)} problems): {agg} -> {out}",
              flush=True)
        summary["mbpp"] = agg
    if args.dataset in ("humaneval", "both"):
        ds = load_dataset("openai/openai_humaneval")
        probs = list(ds["test"])[: args.limit]
        res, agg = evaluate(model, tok, probs, he_prompt, he_passes,
                            args.n, ks, args.max_new, cfg.ctx,
                            args.temperature, "humaneval")
        out = args.ckpt.replace(".pt", "_codeeval_humaneval.json")
        json.dump({"agg": agg, "per_problem": res}, open(out, "w"))
        print(f"HumanEval ({len(probs)} problems): {agg} -> {out}",
              flush=True)
        summary["humaneval"] = agg
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
