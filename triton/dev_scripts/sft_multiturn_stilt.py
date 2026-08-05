"""Multi-turn conversational SFT for Stilt (2026-07-26, user request).

The existing -it SFT already trains on smol-smoltalk multi-turn dialogs
with per-assistant-turn loss, BUT it DROPS any conversation exceeding
ctx (measured 44% of smoltalk, biased toward the LONGEST multi-turn
convos, avg 3.0 assistant turns) — starving exactly the signal that
teaches multi-turn behavior.

This script fixes that with PREFIX-ANCHORED tail-truncation: over-ctx
dialogs keep whole turns from the START until ctx (dropping the newest
overflow) instead of being discarded. Every kept assistant target thus
retains its full preceding context — unlike front-truncation, which
would drop early turns a later target depends on and train the model
to confabulate the answer. Optional --min-assistant-turns concentrates
on genuine multi-turn.

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
    """PREFIX-anchored tail-truncation: keep whole turns from the START
    until ctx, dropping the NEWEST overflow. Every kept assistant turn
    therefore retains its complete preceding context back to turn 0 —
    so we never train a target whose supporting info was truncated away
    (front-truncation would do that and teach confabulation). Cost: we
    can't train on turns deeper than ctx tokens of history — but such
    targets are unlearnable at this ctx anyway."""
    segs = _segments(tok, messages, eot)
    if not segs:
        return None
    kept, acc = [], 0
    for seg in segs:                         # oldest first
        if acc + len(seg[0]) > ctx:
            break                            # drop this and all newer
        kept.append(seg)
        acc += len(seg[0])
    ids = [t for seg in kept for t in seg[0]]
    mask = [t for seg in kept for t in seg[1]]
    if len(ids) < 8 or sum(mask) == 0:       # need >=1 complete-context target
        return None
    return (np.asarray(ids, dtype=np.int64),
            np.asarray(mask, dtype=np.bool_))


def build_tensors(tok, ctx, min_asst, identity_path, upsample,
                  max_examples=200_000, dataset="smol", seed=0):
    eot = tok.eos_token_id
    xs, masks = [], []
    if dataset == "full":
        # full smoltalk (all sources, incl. single-turn); seeded shuffle
        # so the sample is source-balanced, not prefix-biased
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
        ds = ds.shuffle(seed=seed)
    else:
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
          f"{trunc} tail-truncated to a complete-context prefix, kept "
          f"not dropped), identity {n_id}" % min_asst, flush=True)
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
    ap.add_argument("--ctx", type=int, default=None,
                    help="encoding/training context override (NoPE models "
                         "train beyond their pretrain ctx; default cfg.ctx)")
    ap.add_argument("--dataset", choices=["smol", "full"], default="smol",
                    help="'full' = HuggingFaceTB/smoltalk (all sources)")
    ap.add_argument("--max-examples", type=int, default=200_000)
    ap.add_argument("--out-suffix", default="-mt",
                    help="ckpt name suffix (use -it when this IS the it)")
    ap.add_argument("--token-cap", type=int, default=24576,
                    help="max padded tokens per micro-batch (long convos "
                         "split the step; loss renormalized globally)")
    args = ap.parse_args()

    blob = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    tok = AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(args.seed)

    eff_ctx = args.ctx or cfg.ctx
    if args.ctx and args.ctx != cfg.ctx:
        assert getattr(cfg, "nope", False) or getattr(cfg, "rope", False), \
            "--ctx beyond pretrain requires a position-free model"
    xs, masks = build_tensors(tok, eff_ctx, args.min_asst,
                              args.identity, args.identity_upsample,
                              args.max_examples, args.dataset, args.seed)
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
        tgt_full = m[:, 1:]
        n_tgt = tgt_full.sum().clamp_min(1).float()
        opt.zero_grad(set_to_none=True)
        # split rows into micro-batches under a padded-token cap (long
        # convos at large --ctx OOM a full padded batch); loss summed per
        # micro-batch and normalized by the step's total target count
        B, L = x.shape
        rows_per_mb = max(1, args.token_cap // L)
        loss_val = 0.0
        for i in range(0, B, rows_per_mb):
            x_ = x[i:i + rows_per_mb]
            t_ = tgt_full[i:i + rows_per_mb]
            if t_.sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x_)
                mb = F.cross_entropy(logits[:, :-1][t_], x_[:, 1:][t_],
                                     reduction="sum") / n_tgt
            mb.backward()
            loss_val += mb.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.log({"step": step, "sft_loss": loss_val,
                 "grad_norm": float(gn), "lr": sched.get_last_lr()[0]})
        if step % 50 == 0:
            print(f"step {step}/{total_steps} loss {loss.item():.4f}",
                  flush=True)

    out = args.base_ckpt.replace(".pt", args.out_suffix + ".pt")
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
                        lg, _ = model(cur[:, -eff_ctx:])
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
