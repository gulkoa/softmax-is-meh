#!/usr/bin/env python3
"""Where is the logit-growth probe's max |score|? (descriptive, CPU)

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

    python scripts/probe_locate.py CKPT [CKPT ...] [--entropy] [--heads L ...]
                                   [--signature] [--trainer PATH] [--threads N]

--heads L prints layer L (1-based) per head: median |q| and |k|, how much
of q and k is one vector shared by every position (|mean| / mean |.|; 1 =
identical at every position), the cosine between the two mean vectors,
max |score|, median and max within-row spread, and the mean entropy and
top-1 weight over query positions 512-1,023 from fp32 scores.

--signature counts, per layer, the heads with the q/k COLLAPSE signature
(defined 2026-10-16 from the softmax probe run's step-3,000 checkpoint,
thesis/experiments/2026-10-16-qk-collapse-survey.md):
    shared_q >= 0.99 and shared_k >= 0.99 and cos(qbar, kbar) >= 0.95
    and median|q| * median|k| / sqrt(hd) >= 1,000
and prints one SIGNATURE line per checkpoint.

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
import os
import sys

import numpy as np
import torch

TRAINER = ("/users/PAS2402/alexg/softmax/.worktrees/qknorm/triton/"
           "dev_scripts/train_gpt2_stieltjes.py")
SIG = dict(shared=0.99, cos=0.95, scale=1000.0)


def stj_ref(q, k, v, causal=True, sm_scale=1.0, stieltjes_q=4.0, **_):
    """Normalized Stieltjes weights by Newton iteration (as the probe)."""
    s = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    return (weights(s, "stj", stieltjes_q) @ v.float()).to(v.dtype)


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


def head_stats(qf, kf):
    """Per head (H,): median norms, shared fractions, cosine of the means."""
    qn, kn = qf.norm(dim=-1), kf.norm(dim=-1)             # (H, S)
    qb, kb = qf.mean(1), kf.mean(1)                       # (H, D)
    return {"mq": qn.median(-1).values, "mk": kn.median(-1).values,
            "shq": qb.norm(dim=-1) / qn.mean(-1),
            "shk": kb.norm(dim=-1) / kn.mean(-1),
            "cos": (qb * kb).sum(-1) / (qb.norm(dim=-1) * kb.norm(dim=-1))}


def collapsed(st, hd):
    return ((st["shq"] >= SIG["shared"]) & (st["shk"] >= SIG["shared"])
            & (st["cos"] >= SIG["cos"])
            & (st["mq"] * st["mk"] / math.sqrt(hd) >= SIG["scale"]))


def heads(qf, kf, L, args):
    H, S, D = qf.shape
    s = qf @ kf.transpose(-2, -1) / math.sqrt(D)
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    p = weights(s, args.attn, args.stj_q)
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
    spread = (s.masked_fill(~causal, float("-inf")).amax(-1)
              - s.masked_fill(~causal, float("inf")).amin(-1))[:, 1:]
    st = head_stats(qf, kf)
    print(f"layer {L} per head: med|q| med|k| shared_q shared_k cos(qbar,kbar) "
          "max|s| med_spread max_spread ent(512+) top1(512+)")
    for h in range(H):
        print(f"  h{h:<2} {st['mq'][h].item():8.1f} {st['mk'][h].item():8.1f} "
              f"{st['shq'][h].item():6.3f} {st['shk'][h].item():6.3f} "
              f"{st['cos'][h].item():7.3f} "
              f"{s[h].masked_fill(~causal, 0).abs().max().item():10.1f} "
              f"{spread[h].median().item():9.1f} {spread[h].max().item():9.1f} "
              f"{ent[h, S // 2:].mean().item():7.3f} "
              f"{p[h, S // 2:].amax(-1).mean().item():6.3f}")


def analyse(T, path, a, probe_stats):
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    args = argparse.Namespace(**ck["args"])
    model = T.GPT(args)
    model.load_state_dict({k.replace("_orig_mod.", ""): v
                           for k, v in ck["model"].items()})
    model.eval()
    x, _ = T.Shards("val", args.data_mix).batch(
        1, args.ctx, np.random.default_rng(777), "cpu")
    T.stieltjes_attention = stj_ref
    cap = []
    T.attn_probe_stats = lambda q, k, *r: cap.append((q, k)) or {}
    for b in model.blocks:
        b.attn._probe = {}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(x)
    hd = args.n_embd // args.n_head
    name = os.path.basename(path)
    print(f"{name}: step {ck['step']}, attn {args.attn}")
    print("layer   max|s|  head  row   col  sign | max|s| rows>0 | "
          "max row spread (head,row) | |q| |k| at max vs median"
          + (" | collapsed heads, max med|q||k|/sqrt(hd)" if a.signature else ""))
    sig = []
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
        extra = ""
        if a.signature:
            st = head_stats(qf, kf)
            n = int(collapsed(st, hd).sum())
            sig.append(n)
            extra = f" | {n:2d} {(st['mq'] * st['mk']).max().item() / math.sqrt(hd):10.1f}"
        print(f"L{i + 1:<3} {sa.max().item():10.1f} {h:4d} {r:5d} {c:5d} "
              f"{'+' if s[h, r, c] > 0 else '-':>4} | {m_rows:10.1f} | "
              f"{spread.max().item():10.1f} ({sh},{sr + 1}) | "
              f"{qn[h, r].item():.1f} {kn[h, c].item():.1f} vs "
              f"{qn[h].median().item():.1f} {kn[h].median().item():.1f}{extra}")
    if a.signature:
        print(f"SIGNATURE {name} step {ck['step']}: collapsed heads per layer "
              f"{sig}, total {sum(sig)}")
    for L in a.heads:
        heads(cap[L - 1][0].float()[0], cap[L - 1][1].float()[0], L, args)
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
    print(flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--trainer", default=TRAINER)
    ap.add_argument("--entropy", action="store_true")
    ap.add_argument("--heads", type=int, nargs="*", default=[])
    ap.add_argument("--signature", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    spec = importlib.util.spec_from_file_location("pilot", a.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    probe_stats = T.attn_probe_stats
    for path in a.ckpts:
        analyse(T, path, a, probe_stats)


if __name__ == "__main__":
    sys.exit(main())
