"""-it v2 SFT: smol-smoltalk WITH system-turn support + identity mix.

v1 (sft_stilt_it.py, frozen under a queued job) silently dropped
smoltalk's system messages, so shipped -it models ignore system
prompts. v2 template adds a plain-text system marker:

    <|system|>\n{...}\n<|user|>\n{...}\n<|assistant|>\n{...}<|eot|>

Loss on assistant tokens only. Identity/system-following rows
(gen_identity_data.py) are mixed in upsampled. Saves ckpt_<label>-it2.pt.

Usage: python sft_stilt_it2.py <base_ckpt.pt> --identity identity.json
           [--identity-upsample 20] [--tokens 2.5e8] ...
"""
import argparse
import json
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
os.environ.setdefault("WANDB_MODE", "online")
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import wandb  # noqa: E402

DEVICE = torch.device("cuda")
S, U, A = "<|system|>\n", "<|user|>\n", "<|assistant|>\n"


def encode_dialog(tok, messages, ctx, eot):
    ids, mask = [], []
    for m in messages:
        if m["role"] == "system":
            t = tok(S + m["content"] + "\n",
                    add_special_tokens=False).input_ids
            ids += t
            mask += [0] * len(t)
        elif m["role"] == "user":
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
        return None
    return (np.asarray(ids, dtype=np.int64),
            np.asarray(mask, dtype=np.bool_))


def build_tensors(tok, ctx, identity_path, upsample, max_examples=200_000):
    eot = tok.eos_token_id
    xs, masks = [], []
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    n_sys = 0
    for ex in ds.select(range(min(max_examples, len(ds)))):
        enc = encode_dialog(tok, ex["messages"], ctx, eot)
        if enc:
            xs.append(enc[0])
            masks.append(enc[1])
            if any(m["role"] == "system" for m in ex["messages"]):
                n_sys += 1
    n_id = 0
    if identity_path:
        rows = json.load(open(identity_path))
        for _ in range(upsample):
            for r in rows:
                enc = encode_dialog(tok, r["messages"], ctx, eot)
                if enc:
                    xs.append(enc[0])
                    masks.append(enc[1])
                    n_id += 1
    print(f"SFT v2 examples: {len(xs)} (smoltalk-with-system {n_sys}, "
          f"identity x{upsample} = {n_id})", flush=True)
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
    ap.add_argument("base_ckpt")
    ap.add_argument("--identity", default=None)
    ap.add_argument("--identity-upsample", type=int, default=20)
    ap.add_argument("--tokens", type=float, default=2.5e8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.base_ckpt, map_location="cpu",
                      weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    tok = AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(args.seed)

    xs, masks = build_tensors(tok, cfg.ctx, args.identity,
                              args.identity_upsample)
    avg_len = float(np.mean([len(x) for x in xs]))
    total_steps = int(args.tokens // (args.bs * avg_len))
    print(f"SFT v2: {total_steps} steps (avg len {avg_len:.0f})",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    rng = np.random.default_rng(args.seed)
    gen = batches(xs, masks, args.bs, rng, tok.eos_token_id)

    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"sftv2-{os.path.basename(args.base_ckpt)}"
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
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.log({"step": step, "sft_loss": loss.item(),
                 "grad_norm": float(gnorm),
                 "lr": sched.get_last_lr()[0]})
        if step % 50 == 0:
            print(f"step {step:5d}/{total_steps} loss {loss.item():.4f}",
                  flush=True)

    out = args.base_ckpt.replace(".pt", "-it2.pt")
    torch.save({"model": model.state_dict(), "args": vars(cfg),
                "sftv2_args": vars(args)}, out)
    print(f"saved {out}", flush=True)

    # greedy identity + system-following demos
    model.eval()
    demos = [
        (S + "You are stilt, a helpful AI assistant\n"
         + U + "Who are you?\n" + A),
        (S + "Respond only with 'yes' or 'no'.\n"
         + U + "Is water wet?\n" + A),
        (U + "What is the capital of France?\n" + A),
    ]
    for p in demos:
        ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        cur = ids
        with torch.no_grad():
            for _ in range(60):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(cur[:, -cfg.ctx:])
                nxt = logits[0, -1].argmax()
                if nxt.item() == tok.eos_token_id:
                    break
                cur = torch.cat([cur, nxt[None, None]], 1)
        print(f"PROMPT {p!r}\n  -> "
              f"{tok.decode(cur[0, ids.shape[1]:])!r}", flush=True)
    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
