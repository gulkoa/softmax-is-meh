"""Improvement #2: instruction-tune a Stilt base checkpoint -> `-it`.

Data: HuggingFaceTB/smol-smoltalk (curated for sub-1B models). Plain-text
chat template (GPT-2 BPE has no chat tokens):
    <|user|>\n{...}\n<|assistant|>\n{...}<|endoftext|> per turn pair.
Loss on assistant tokens only. Short cosine SFT, checkpoint saved as
ckpt_<label>-it.pt next to the base.

Usage: python sft_stilt_it.py <base_ckpt.pt> [--epochs-tokens 3e8] ...
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

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
os.environ.setdefault("WANDB_MODE", "offline")
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import wandb  # noqa: E402

DEVICE = torch.device("cuda")
U, A = "<|user|>\n", "<|assistant|>\n"


def build_sft_tensors(tok, ctx, max_examples=200_000):
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    xs, masks = [], []
    eot = tok.eos_token_id
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ids, mask = [], []
        for m in ex["messages"]:
            if m["role"] == "user":
                t = tok(U + m["content"] + "\n",
                        add_special_tokens=False).input_ids
                ids += t
                mask += [0] * len(t)
            elif m["role"] == "assistant":
                pre = tok(A, add_special_tokens=False).input_ids
                body = tok(m["content"], add_special_tokens=False).input_ids
                ids += pre + body + [eot]
                mask += [0] * len(pre) + [1] * (len(body) + 1)
        if len(ids) < 8 or len(ids) > ctx:
            continue
        xs.append(np.asarray(ids, dtype=np.int64))
        masks.append(np.asarray(mask, dtype=np.bool_))
    print(f"SFT examples: {len(xs)}", flush=True)
    return xs, masks


def batches(xs, masks, bs, ctx, rng, pad_id):
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
    ap.add_argument("base_ckpt")
    ap.add_argument("--tokens", type=float, default=2.5e8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    tok = AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(args.seed)

    xs, masks = build_sft_tensors(tok, cfg.ctx)
    avg_len = float(np.mean([len(x) for x in xs]))
    total_steps = int(args.tokens // (args.bs * avg_len))
    print(f"SFT: {total_steps} steps (avg len {avg_len:.0f})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    rng = np.random.default_rng(args.seed)
    gen = batches(xs, masks, args.bs, cfg.ctx, rng, tok.eos_token_id)

    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"sft-{os.path.basename(args.base_ckpt)}"
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
        if step % 50 == 0:
            print(f"step {step:5d}/{total_steps} loss {loss.item():.4f}",
                  flush=True)
            run.log({"step": step, "sft_loss": loss.item()})

    out = args.base_ckpt.replace(".pt", "-it.pt")
    torch.save({"model": model.state_dict(), "args": vars(cfg),
                "sft_args": vars(args)}, out)
    print(f"saved {out}", flush=True)

    # sample dialogues
    model.eval()
    for qtext in ["What is the capital of France?",
                  "Explain photosynthesis in one sentence.",
                  "Write a haiku about the sea."]:
        ids = tok(U + qtext + "\n" + A,
                  return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            cur = ids
            for _ in range(60):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(cur[:, -cfg.ctx:])
                nxt = logits[0, -1].argmax()
                if nxt.item() == tok.eos_token_id:
                    break
                cur = torch.cat([cur, nxt[None, None]], 1)
        print(f"Q: {qtext}\nA: "
              f"{tok.decode(cur[0, ids.shape[1]:])!r}\n", flush=True)
    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
