#!/usr/bin/env python3
"""Where is the logit-growth probe's max |score|? (descriptive, login node, CPU)

The probe (thesis/experiments/2026-10-15-logit-growth-probe.md) logs only
M = max over layers of max |score| on one fixed validation sequence. Both
mappings are shift-invariant per query row (softmax exactly; Stieltjes
because lambda shifts with the row), so a huge |score| matters only if the
row's scores also SPREAD. This rebuilds the probe pass from a checkpoint
with the frozen pilot trainer's model, then reports per layer: where the
max |score| sits (head, query row, key column, sign), the max |score|
without query row 0 (which attends only to itself), the largest
within-row spread max_j s_ij - min_j s_ij, and the norms of the q and k
behind the max against the layer's median norms.

    python scripts/probe_locate.py CKPT [--trainer PATH] [--entropy] [--heads L ...]

--heads L prints layer L (1-based) per head: median |q| and |k|, how much
of q and k is one vector shared by every position (|mean| / mean |.|; 1 =
identical at every position), the cosine between the two mean vectors,
max |score|, median and max within-row spread, and the mean entropy and
top-1 weight over query positions 512-1,023 from fp32 scores.

--entropy also prints each layer's min-head entropy two ways, with the
trainer's own probe function: from bf16 scores, as the training probe logs
it (it runs under bf16 autocast, which casts its q @ k^T to bf16), and from
fp32 scores, as the attention kernels compute them. Where |score| is in
the thousands, bf16 rounds whole rows of scores to a few levels and the
logged entropy is too high (slow on CPU for Stieltjes: ~10 min).

Stieltjes checkpoints run with a PyTorch reference mapping in place of the
Triton kernel (CPU), so only the attention OUTPUT differs slightly from
training; the reported scores come from the same q and k computation.
"""
import argparse
import importlib.util
import math
import sys

import numpy as np
import torch

TRAINER = ("/users/PAS2402/alexg/softmax/.worktrees/qknorm/triton/"
           "dev_scripts/train_gpt2_stieltjes.py")


def stj_ref(q, k, v, causal=True, sm_scale=1.0, stieltjes_q=4.0, **_):
    """Normalized Stieltjes weights by Newton iteration (as the probe)."""
    s = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    S = s.shape[-1]
    mask = torch.ones(S, S, dtype=torch.bool).tril()
    s = s.masked_fill(~mask, float("-inf"))
    x = s - s.amax(-1, keepdim=True)
    lam = torch.full_like(x[..., :1], 1.1)
    for _ in range(30):
        d = (lam - x).clamp_min(1e-6)
        f = d.pow(-stieltjes_q).sum(-1, keepdim=True) - 1.0
        fp = -stieltjes_q * d.pow(-stieltjes_q - 1.0).sum(-1, keepdim=True)
        lam = lam - f / fp
    p = (lam - x).clamp_min(1e-6).pow(-stieltjes_q)
    p = p / p.sum(-1, keepdim=True)
    return (p @ v.float()).to(v.dtype)


def weights(s, attn, stj_q):
    """Attention weights from fp32 scores (B, H, S, S), as the kernels compute."""
    S = s.shape[-1]
    s = s.masked_fill(~torch.ones(S, S, dtype=torch.bool).tril(), float("-inf"))
    if attn == "sdpa":
        return torch.softmax(s, -1)
    x = s - s.amax(-1, keepdim=True)
    lam = torch.full_like(x[..., :1], 1.1)
    for _ in range(30):
        d = (lam - x).clamp_min(1e-6)
        lam = lam - (d.pow(-stj_q).sum(-1, keepdim=True) - 1.0) / (
            -stj_q * d.pow(-stj_q - 1.0).sum(-1, keepdim=True))
    p = (lam - x).clamp_min(1e-6).pow(-stj_q)
    return p / p.sum(-1, keepdim=True)


def heads(q, k, L, args, _):
    qf, kf = q.float()[0], k.float()[0]
    H, S, D = qf.shape
    s = qf @ kf.transpose(-2, -1) / math.sqrt(D)
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    p = weights(s, args.attn, args.stj_q)
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
    spread = (s.masked_fill(~causal, float("-inf")).amax(-1)
              - s.masked_fill(~causal, float("inf")).amin(-1))[:, 1:]
    print(f"layer {L} per head: med|q| med|k| shared_q shared_k cos(qbar,kbar) "
          "max|s| med_spread max_spread ent(512+) top1(512+)")
    for h in range(H):
        qb, kb = qf[h].mean(0), kf[h].mean(0)
        print(f"  h{h:<2} {qf[h].norm(dim=-1).median().item():8.1f} "
              f"{kf[h].norm(dim=-1).median().item():8.1f} "
              f"{(qb.norm() / qf[h].norm(dim=-1).mean()).item():6.3f} "
              f"{(kb.norm() / kf[h].norm(dim=-1).mean()).item():6.3f} "
              f"{(qb @ kb / (qb.norm() * kb.norm())).item():7.3f} "
              f"{s[h].masked_fill(~causal, 0).abs().max().item():10.1f} "
              f"{spread[h].median().item():9.1f} {spread[h].max().item():9.1f} "
              f"{ent[h, S // 2:].mean().item():7.3f} "
              f"{p[h, S // 2:].amax(-1).mean().item():6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--trainer", default=TRAINER)
    ap.add_argument("--entropy", action="store_true")
    ap.add_argument("--heads", type=int, nargs="*", default=[])
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("pilot", a.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    ck = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False)
    args = argparse.Namespace(**ck["args"])
    model = T.GPT(args)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.eval()
    x, _ = T.Shards("val", args.data_mix).batch(
        1, args.ctx, np.random.default_rng(777), "cpu")
    if args.attn == "stj":
        T.stieltjes_attention = stj_ref
    probe_stats = T.attn_probe_stats
    cap = []
    T.attn_probe_stats = lambda q, k, *r: cap.append((q, k)) or {}
    for b in model.blocks:
        b.attn._probe = {}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(x)
    hd = args.n_embd // args.n_head
    print(f"{a.ckpt.split('/')[-1]}: step {ck['step']}, attn {args.attn}")
    print("layer   max|s|  head  row   col  sign | max|s| rows>0 | "
          "max row spread (head,row) | |q| |k| at max vs median")
    for i, (q, k) in enumerate(cap):
        qf, kf = q.float()[0], k.float()[0]               # (H, S, D)
        s = qf @ kf.transpose(-2, -1) / math.sqrt(hd)     # (H, S, S)
        S = s.shape[-1]
        causal = torch.ones(S, S, dtype=torch.bool).tril()
        sa = s.masked_fill(~causal, 0.0).abs()
        h, r, c = np.unravel_index(int(sa.argmax()), sa.shape)
        m_rows = sa[:, 1:].amax().item()
        hi = s.masked_fill(~causal, float("-inf")).amax(-1)
        lo = s.masked_fill(~causal, float("inf")).amin(-1)
        spread = (hi - lo)[:, 1:]                         # row 0 has one key
        sh, sr = np.unravel_index(int(spread.argmax()), spread.shape)
        qn, kn = qf.norm(dim=-1), kf.norm(dim=-1)         # (H, S)
        print(f"L{i + 1:<3} {sa.max().item():10.1f} {h:4d} {r:5d} {c:5d} "
              f"{'+' if s[h, r, c] > 0 else '-':>4} | {m_rows:10.1f} | "
              f"{spread.max().item():10.1f} ({sh},{sr + 1}) | "
              f"{qn[h, r].item():.1f} {kn[h, c].item():.1f} vs "
              f"{qn[h].median().item():.1f} {kn[h].median().item():.1f}")
    for L in a.heads:
        heads(*cap[L - 1], L, args, probe_stats)
    if a.entropy:
        bf, fp = [], []
        for q, k in cap:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                bf.append(probe_stats(q, k, args.attn, args.stj_q, hd))
            with torch.no_grad():
                fp.append(probe_stats(q, k, args.attn, args.stj_q, hd))
        print("min-head entropy, bf16 scores (as logged):",
              " ".join(f"{d['min_head_entropy']:.3f}" for d in bf))
        print("min-head entropy, fp32 scores (as kernel):",
              " ".join(f"{d['min_head_entropy']:.3f}" for d in fp))


if __name__ == "__main__":
    sys.exit(main())
