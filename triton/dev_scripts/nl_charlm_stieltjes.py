"""
NATURAL LANGUAGE experiment #1: character-level LM (tinyshakespeare) with
Stieltjes flash attention vs softmax (SDPA flash).

Model: GPT-mini, NoPE (no positional embeddings — matches the project's
length-extrapolation line and lets us eval at longer contexts than trained).
6 layers, 8 heads, d=512 (head_dim 64), trained at block=512.

Arms (--attn):
  sdpa    : softmax via F.scaled_dot_product_attention (flash), causal
  flashn  : Triton Stieltjes kernel, normalize=True, causal, bf16, num_iter=8

Metrics: train/val cross-entropy (nats/char) at the training context, plus
zero-shot val loss at context {512, 1024, 2048, 4096} (NoPE extrapolation).
wandb-logged incrementally (offline on compute nodes).
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

DATA_PATHS = {
    "shakespeare": "/users/PAS2402/alexg/softmax/softmax-is-meh/results/nl_data/shakespeare.txt",
    # 100MB GitHub Python (codeparrot-clean fallback for gated the-stack-smol;
    # same distribution family as The Stack python subset). Byte-level.
    "stack": "/users/PAS2402/alexg/softmax/softmax-is-meh/results/nl_data/stack_python.txt",
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        if cfg.attn == "asflashn":
            # AS-Stieltjes: learnable per-position log-length scaling
            # s_i = 1 + softplus(w_beta . q_i) * (log(i+1))^gamma, folded into Q.
            self.w_beta = nn.Parameter(torch.zeros(self.n_head, self.head_dim))
            self._log_gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        B, S, E = x.shape
        q, k, v = self.qkv(x).split(E, dim=2)
        q = q.view(B, S, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        if self.cfg.attn == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # flashn / asflashn
            if self.cfg.attn == "asflashn":
                beta = F.softplus(torch.einsum("bhsd,hd->bhs", q, self.w_beta))
                logn = torch.log(torch.arange(1, S + 1, device=q.device,
                                              dtype=torch.float32).clamp(min=2.0))
                scale = 1.0 + beta * logn.pow(self._log_gamma.exp())
                q = (q * scale.unsqueeze(-1).to(q.dtype)).contiguous()
            o = stieltjes_attention(
                q, k, v, causal=True,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                stieltjes_q=self.cfg.stieltjes_q, num_iter=8, normalize=True)
        o = o.transpose(1, 2).reshape(B, S, E)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attn(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CharGPT(nn.Module):
    """NoPE decoder-only char LM."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        self.head.weight = self.emb.weight  # tied

    def forward(self, idx, targets=None):
        x = self.emb(idx)                    # NoPE: no positional embedding
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(which="shakespeare"):
    path = DATA_PATHS[which]
    if which == "stack":
        # byte-level LM: vocab 256, no char-set pass over 100MB
        data = np.fromfile(path, dtype=np.uint8).astype(np.int64)
        vocab = 256
    else:
        text = open(path, encoding="utf-8").read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = np.array([stoi[c] for c in text], dtype=np.int64)
        vocab = len(chars)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab


def get_batch(data, block, bs, device, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + block].copy()) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block + 1].copy()) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, data, block, device, iters=40, bs=8, seed=999):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block, bs, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["sdpa", "flashn", "asflashn"],
                    required=True)
    ap.add_argument("--data", choices=["shakespeare", "stack"],
                    default="shakespeare")
    ap.add_argument("--q", type=float, default=4.0, dest="stieltjes_q")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-blocks", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096])
    args = ap.parse_args()

    device = torch.device("cuda")
    train_d, val_d, vocab = load_data(args.data)
    args.vocab = vocab
    torch.manual_seed(args.seed)

    label = (f"{args.attn}" if args.attn == "sdpa"
             else f"{args.attn}-q{args.stieltjes_q:g}")
    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"nl-{args.data}-{label}-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config={**vars(args), "vocab": vocab, "pos_emb": "NoPE"},
    )
    print(f"data={args.data} arm={label} vocab={vocab} "
          f"train={len(train_d)} val={len(val_d)}")

    model = CharGPT(args).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params: {nparam/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.1)
    gen = torch.Generator().manual_seed(args.seed * 7919 + 1)

    model.train()
    for it in range(args.iters):
        x, y = get_batch(train_d, args.block, args.bs, device, gen)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 100 == 0:
            vl = eval_loss(model, val_d, args.block, device, iters=10)
            print(f"it {it:5d} train {loss.item():.4f} val {vl:.4f}", flush=True)
            run.log({"iter": it, "train_loss": loss.item(), "val_loss": vl})

    # Final: val loss across context lengths (NoPE zero-shot extrapolation)
    print("\nfinal val loss by context length:")
    for blk in args.eval_blocks:
        if blk > len(val_d) - 2:
            continue
        vl = eval_loss(model, val_d, blk, device,
                       iters=40 if blk <= 1024 else 16,
                       bs=max(1, 8 * args.block // blk))
        bpc = vl / math.log(2)
        print(f"  ctx {blk:5d}: val_loss {vl:.4f}  ({bpc:.3f} bits/char)")
        run.log({f"final/val_loss_ctx{blk}": vl, f"final/bpc_ctx{blk}": bpc})
        run.summary[f"val_loss_ctx{blk}"] = vl

    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
