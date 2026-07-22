"""Stage-1 code-SFT warm-start for the RL line (plan: thesis/findings/
2026-07-20-plan-rl-coding-stilt11.md): fine-tune an -it checkpoint on
MBPP-train canonical solutions (+ optional synthetic tasks with
reference solutions), in EXACTLY the GRPO prompt format (build_prompt,
2 shown tests) with fenced-code-block targets — so the policy enters RL
in-distribution and extract_code() is reliable.

Usage: python sft_code_stilt.py <it_ckpt.pt> [--synthetic tasks.json]
           [--tokens 4e5] [--bs 16] [--lr 2e-5]
Saves ckpt_<label>-code.pt next to the input.
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT, FW_DIR  # noqa: E402
from grpo_code_stilt import build_prompt  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
os.environ.setdefault("WANDB_MODE", "offline")
import json  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import wandb  # noqa: E402

DEVICE = torch.device("cuda")


def build_code_tensors(tok, ctx, synthetic_path=None):
    ds = load_dataset("google-research-datasets/mbpp", "full")
    problems = list(ds["train"])
    if synthetic_path:
        problems += json.load(open(synthetic_path))
    xs, masks = [], []
    eot = tok.eos_token_id
    skipped = 0
    for p in problems:
        prompt = build_prompt(p)
        target = "```python\n" + p["code"].strip() + "\n```"
        pids = tok(prompt, add_special_tokens=False).input_ids
        tids = tok(target, add_special_tokens=False).input_ids + [eot]
        ids = pids + tids
        if len(ids) > ctx:
            skipped += 1
            continue
        mask = [0] * len(pids) + [1] * len(tids)
        xs.append(np.asarray(ids, dtype=np.int64))
        masks.append(np.asarray(mask, dtype=np.bool_))
    print(f"code-SFT examples: {len(xs)} (skipped {skipped} over-ctx)",
          flush=True)
    return xs, masks


def batches(xs, masks, bs, rng, pad_id):
    while True:
        idx = rng.integers(0, len(xs), size=bs)
        L = max(len(xs[i]) for i in idx)
        x = np.full((bs, L), pad_id, dtype=np.int64)
        m = np.zeros((bs, L), dtype=np.bool_)
        for j, i in enumerate(idx):
            x[j, :len(xs[i])] = xs[i]
            m[j, :len(xs[i])] = masks[i]
        yield (torch.from_numpy(x).to(DEVICE),
               torch.from_numpy(m).to(DEVICE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("it_ckpt")
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--tokens", type=float, default=4e5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.it_ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    tok = AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(args.seed)

    xs, masks = build_code_tensors(tok, cfg.ctx, args.synthetic)
    avg_len = float(np.mean([len(x) for x in xs]))
    total_steps = max(1, int(args.tokens // (args.bs * avg_len)))
    print(f"code-SFT: {total_steps} steps (avg len {avg_len:.0f})",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    rng = np.random.default_rng(args.seed)
    gen = batches(xs, masks, args.bs, rng, tok.eos_token_id)

    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"codesft-{os.path.basename(args.it_ckpt)}"
                          f"-{os.environ.get('SLURM_JOB_ID', 'local')}",
                     config={**vars(args), "base_args": blob["args"]})
    model.train()
    for step in range(total_steps):
        x, m = next(gen)
        tgt_mask = m[:, 1:]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            lg = logits[:, :-1][tgt_mask]
            loss = F.cross_entropy(lg, x[:, 1:][tgt_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0:
            print(f"step {step:4d}/{total_steps} loss {loss.item():.4f}",
                  flush=True)
            run.log({"step": step, "codesft_loss": loss.item()})

    out = args.it_ckpt.replace(".pt", "-code.pt")
    torch.save({"model": model.state_dict(), "args": vars(cfg),
                "codesft_args": vars(args)}, out)
    print(f"saved {out}", flush=True)

    # greedy sample on one held-out-style prompt
    model.eval()
    demo = {"text": "Write a function to find the sum of the squares of "
                    "the first n natural numbers.",
            "test_list": ["assert sum_sq(3) == 14", "assert sum_sq(1) == 1"]}
    ids = tok(build_prompt(demo), return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        cur = ids
        for _ in range(120):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(cur[:, -cfg.ctx:])
            nxt = logits[0, -1].argmax()
            if nxt.item() == tok.eos_token_id:
                break
            cur = torch.cat([cur, nxt[None, None]], 1)
    print("DEMO:", repr(tok.decode(cur[0, ids.shape[1]:])), flush=True)
    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
