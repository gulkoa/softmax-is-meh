"""Multi-turn conversational SFT for Stilt (2026-07-26, user request).

The existing -it SFT already trains on smol-smoltalk multi-turn dialogs
with per-assistant-turn loss, BUT it DROPS any conversation exceeding
ctx (measured 44% of smoltalk, biased toward the LONGEST multi-turn
convos, avg 3.0 assistant turns) — starving exactly the signal that
teaches multi-turn behavior.

This script fixes that with END-ANCHORED front-truncation: over-ctx
dialogs keep their most-recent whole turns (dropping oldest first,
starting at a user/system boundary) instead of being discarded — so a
10-turn conversation still contributes its later turns. Optional
--min-assistant-turns concentrates on genuine multi-turn.

Loss on all assistant turns. Saves ckpt_<label>-mt.pt.

Usage: python sft_multiturn_stilt.py <base_ckpt.pt>
    [--min-assistant-turns 2] [--identity id.json] [--tokens 3e8]
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


def _segments(tok, messages, eot):
    """Per-message (ids, mask, role); assistant turns masked for loss."""
    segs = []
    for m in messages:
        r = m["role"]
        if r == "system":
            t = tok(S + m["content"] + "\n", add_special_tokens=False).input_ids
            segs.append((t, [0] * len(t), r))
        elif r == "user":
            t = tok(U + m["content"] + "\n", add_special_tokens=False).input_ids
            segs.append((t, [0] * len(t), r))
        elif r == "assistant":
            pre = tok(A, add_special_tokens=False).input_ids
            body = tok(m["content"], add_special_tokens=False).input_ids
            segs.append((pre + body + [eot],
                         [0] * len(pre) + [1] * (len(body) + 1), r))
    return segs


def encode_dialog(tok, messages, ctx, eot):
    """End-anchored: keep the most recent whole turns that fit ctx."""
    segs = _segments(tok, messages, eot)
    if not segs:
        return None
    total = sum(len(s[0]) for s in segs)
    if total <= ctx:
        kept = segs
    else:
        kept, acc = [], 0
        for seg in reversed(segs):           # newest first
            if acc + len(seg[0]) > ctx:
                break
            kept.insert(0, seg)
            acc += len(seg[0])
        # start at a user/system boundary (drop leading assistant targets
        # that would train on a truncated-away context)
        while kept and kept[0][2] == "assistant":
            kept.pop(0)
    ids = [t for seg in kept for t in seg[0]]
    mask = [t for seg in kept for t in seg[1]]
    if len(ids) < 8 or sum(mask) == 0:       # need a real assistant target
        return None
    return (np.asarray(ids, dtype=np.int64),
            np.asarray(mask, dtype=np.bool_))


def build_tensors(tok, ctx, min_asst, identity_path, upsample,
                  max_examples=200_000):
    eot = tok.eos_token_id
    xs, masks = [], []
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    kept_mt = trunc = 0
    for ex in ds.select(range(min(max_examples, len(ds)))):
        na = sum(1 for m in ex["messages"] if m["role"] == "assistant")
        if na < min_asst:
            continue
        raw = sum(len(tok(m["content"]).input_ids) for m in ex["messages"])
        enc = encode_dialog(tok, ex["messages"], ctx, eot)
        if enc is None:
            continue
        xs.append(enc[0])
        masks.append(enc[1])
        kept_mt += 1
        if raw > ctx:
            trunc += 1
    n_id = 0
    if identity_path:
        for _ in range(upsample):
            for r in json.load(open(identity_path)):
                enc = encode_dialog(tok, r["messages"], ctx, eot)
                if enc is not None:
                    xs.append(enc[0])
                    masks.append(enc[1])
                    n_id += 1
    print(f"multiturn SFT: {kept_mt} convos (>=%d asst turns; "
          f"{trunc} front-truncated, kept not dropped), identity {n_id}"
          % min_asst, flush=True)
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
        yield (torch.from_numpy(x).to(DEVICE), torch.from_numpy(m).to(DEVICE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_ckpt")
    ap.add_argument("--min-assistant-turns", type=int, default=2,
                    dest="min_asst")
    ap.add_argument("--identity", default=None)
    ap.add_argument("--identity-upsample", type=int, default=15)
    ap.add_argument("--tokens", type=float, default=3e8)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    tok = AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(args.seed)

    xs, masks = build_tensors(tok, cfg.ctx, args.min_asst,
                              args.identity, args.identity_upsample)
    avg = float(np.mean([len(x) for x in xs]))
    total_steps = int(args.tokens // (args.bs * avg))
    print(f"multiturn SFT: {total_steps} steps (avg len {avg:.0f})",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    rng = np.random.default_rng(args.seed)
    gen = batches(xs, masks, args.bs, rng, tok.eos_token_id)
    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"sftmt-{os.path.basename(args.base_ckpt)}"
                          f"-{os.environ.get('SLURM_JOB_ID', 'local')}",
                     config={**vars(args), "base_args": blob["args"]})
    model.train()
    for step in range(total_steps):
        x, m = next(gen)
        tgt = m[:, 1:]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits[:, :-1][tgt], x[:, 1:][tgt])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.log({"step": step, "sft_loss": loss.item(),
                 "grad_norm": float(gn), "lr": sched.get_last_lr()[0]})
        if step % 50 == 0:
            print(f"step {step}/{total_steps} loss {loss.item():.4f}",
                  flush=True)

    out = args.base_ckpt.replace(".pt", "-mt.pt")
    torch.save({"model": model.state_dict(), "args": vars(cfg),
                "sftmt_args": vars(args)}, out)
    print(f"saved {out}", flush=True)

    # multi-turn coherence demo: 3-turn conversation with back-reference
    model.eval()
    convo = [("system", "You are stilt, a helpful AI assistant"),
             ("user", "My favorite animal is the octopus."),
             ("assistant", None),
             ("user", "Why might that animal be considered intelligent?"),
             ("assistant", None),
             ("user", "What was my favorite animal again?"),
             ("assistant", None)]
    hist = ""
    for role, content in convo:
        if role == "system":
            hist += S + content + "\n"
        elif role == "user":
            hist += U + content + "\n"
        elif content is None:  # generate
            hist += A
            ids = tok(hist, return_tensors="pt").input_ids.to(DEVICE)
            cur = ids
            with torch.no_grad():
                for _ in range(50):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg, _ = model(cur[:, -cfg.ctx:])
                    nx = lg[0, -1].argmax()
                    if nx.item() == tok.eos_token_id:
                        break
                    cur = torch.cat([cur, nx[None, None]], 1)
            resp = tok.decode(cur[0, ids.shape[1]:])
            print(f"  A: {resp!r}", flush=True)
            hist += resp + "<|endoftext|>"
    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
