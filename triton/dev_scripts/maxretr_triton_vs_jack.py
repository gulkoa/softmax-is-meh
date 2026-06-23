"""
Apples-to-apples: the Triton Stieltjes kernel vs Jack's bisection, in Jack's
EXACT MaxRetrievalModel architecture (single-query cross-attention, linear
projections, no MLP embedding).

Both variants share IDENTICAL projection weights (copied), see the IDENTICAL
data stream, and use the same q (Stieltjes order). The only difference is the
attention backend:
  - JACK:   translate_logits = StieltjesBSTransform (a19bb33 bisection, PyTorch)
  - TRITON: the Triton kernel in BS-mode (block_lambda_grad=True), which
            replicates the bisection's detached-lambda + argmax-correction
            gradient. Single query is tiled to Nq=Nk=T; output row 0 is taken.

Stages:
  (0) Re-verify BUG 2 fix on raw scores at T not a multiple of 64.
  (1) Forward equivalence on a fixed batch (untrained, shared weights).
  (2) Gradient equivalence (backward through the full model on the same loss).
  (3) Full training comparison (same seed/data) at ID + OOD lengths.

Run in fp32 throughout (Jack's model is fp32; avoids fp16 near-tie fragility).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

PSM_ROOT = "/users/PAS2402/alexg/softmax/probability-simplex-mappings"
sys.path.insert(0, PSM_ROOT)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")

import types
def _alias(pkg, path):
    m = types.ModuleType(pkg); m.__path__ = [path]; sys.modules[pkg] = m
_alias("asentmax_comp", PSM_ROOT)
_alias("asentmax_comp.mappings", os.path.join(PSM_ROOT, "mappings"))
_alias("asentmax_comp.max_retrieval_architecture",
       os.path.join(PSM_ROOT, "max_retrieval_architecture"))

from asentmax_comp.max_retrieval_architecture.architecture import MaxRetrievalModel
from maxretr_bs_vs_nr import StieltjesBSTransform, _NRMappingEnum
from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref


# ---------------------------------------------------------------------------
# Triton-backed MaxRetrievalModel (same architecture, Triton attention)
# ---------------------------------------------------------------------------

class MaxRetrievalModelTriton(MaxRetrievalModel):
    """Identical to MaxRetrievalModel but computes the Stieltjes attention with
    the Triton kernel (BS-mode). Single query tiled to Nq=Nk=T."""

    def __init__(self, *args, stieltjes_q=4.0, num_iter=20, **kwargs):
        super().__init__(*args, **kwargs)
        self._sq = float(stieltjes_q)
        self._num_iter = int(num_iter)

    def forward(self, x_items, x_query, return_attn=False):
        x_query_unsqueezed = x_query.unsqueeze(-1)
        q = self.q_proj(x_query_unsqueezed)   # (B, 1, d)
        k = self.k_proj(x_items)              # (B, T, d)
        v = self.v_proj(x_items)              # (B, T, d)

        B, T, d = k.shape
        if self.attn_score_scale == "inv_sqrt_d":
            sm_scale = self.d_emb ** -0.5
        else:
            sm_scale = 1.0

        # Tile the single query to T rows so Nq = Nk = T (kernel is square).
        q_t = q.expand(B, T, d).unsqueeze(1).contiguous()  # (B, 1, T, d)
        k_t = k.unsqueeze(1).contiguous()                  # (B, 1, T, d)
        v_t = v.unsqueeze(1).contiguous()                  # (B, 1, T, d)

        o = stieltjes_attention(q_t, k_t, v_t, causal=False, sm_scale=sm_scale,
                                stieltjes_q=self._sq, num_iter=self._num_iter,
                                block_lambda_grad=True)     # (B, 1, T, d)
        z = o[:, 0, 0, :]                                   # (B, d)  rows identical
        out_logits = self.phi(z)
        return out_logits


# ---------------------------------------------------------------------------
# Data + training (mirrors maxretr_bs_vs_nr / Jack's main.py)
# ---------------------------------------------------------------------------

def _set_seeds(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(bs, T, n_classes, device, gen):
    priorities = torch.rand(bs, T, device=device, generator=gen)
    classes = torch.randint(0, n_classes, (bs, T), device=device, generator=gen)
    targets = classes.gather(1, priorities.argmax(1, keepdim=True)).squeeze(1).long()
    items = torch.cat([priorities.unsqueeze(-1),
                       F.one_hot(classes, n_classes).float()], dim=-1)
    queries = torch.rand(bs, 1, device=device, generator=gen)
    return items, queries, targets


def train(model, *, seq_len, n_classes, device, steps, bs, lr, wd, warmup, seed):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                          pct_start=warmup / steps, anneal_strategy="cos")
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for step in tqdm(range(steps), desc="train", leave=False):
        g = torch.Generator(device=device); g.manual_seed(seed * 1_000_003 + step)
        items, queries, targets = make_batch(bs, seq_len, n_classes, device, g)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(items, queries), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def evaluate(model, *, seq_len, n_classes, device, samples, bs, seed=12345):
    model.eval()
    correct = total = 0
    ebs = bs * 2
    for i in range(int(np.ceil(samples / ebs))):
        g = torch.Generator(device=device); g.manual_seed(seed + i)
        items, queries, targets = make_batch(ebs, seq_len, n_classes, device, g)
        preds = model(items, queries).argmax(-1)
        correct += (preds == targets).sum().item(); total += targets.numel()
    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage0_bug2_recheck(device):
    print("=" * 78)
    print("STAGE 0 — BUG 2 fix recheck (Triton vs PyTorch ref, fp32, raw scores)")
    print("  T not a multiple of 64 should now be SMALL (was 0.04-0.75 before fix)")
    print("=" * 78)
    D = 64
    for N in [16, 32, 64, 96]:
        for q in [4.0, 16.0]:
            torch.manual_seed(0)
            qd = torch.randn(2, 4, N, D, device=device)
            kd = torch.randn(2, 4, N, D, device=device)
            vd = torch.randn(2, 4, N, D, device=device)
            sm = 1.0 / D ** 0.5
            ref = stieltjes_attention_ref(qd, kd, vd, sm, causal=False,
                                          stieltjes_q=q, num_iter=25, eps=1e-6)
            tri = stieltjes_attention(qd, kd, vd, causal=False, sm_scale=sm,
                                      stieltjes_q=q, num_iter=25)
            err = (tri - ref).abs().max().item()
            print(f"  N={N:3d} (mult64={N%64==0!s:5s}) q={q:>4.0f}: fwd max_err={err:.3e}")
    print()


def build_pair(device, *, d_emb, n_classes, item_dim, sq, num_iter):
    """Build Jack-bisection model and Triton model with IDENTICAL weights."""
    _set_seeds(0)
    model_jack = MaxRetrievalModel(
        simplex_mapping=_NRMappingEnum(StieltjesBSTransform),
        d_emb=d_emb, n_classes=n_classes, item_input_dim=item_dim,
        query_input_dim=1, attn_score_scale="inv_sqrt_d", q=sq,
    ).to(device)
    _set_seeds(0)
    model_tri = MaxRetrievalModelTriton(
        simplex_mapping=_NRMappingEnum(StieltjesBSTransform),
        d_emb=d_emb, n_classes=n_classes, item_input_dim=item_dim,
        query_input_dim=1, attn_score_scale="inv_sqrt_d", q=sq,
        stieltjes_q=sq, num_iter=num_iter,
    ).to(device)
    # Copy projection weights so both start identical.
    model_tri.load_state_dict(model_jack.state_dict(), strict=False)
    return model_jack, model_tri


def stage1_2_fwd_grad(device, *, d_emb, n_classes, item_dim, sq, num_iter):
    print("=" * 78)
    print(f"STAGE 1+2 — forward & gradient equivalence in MaxRetrievalModel (q={sq})")
    print("=" * 78)
    model_jack, model_tri = build_pair(device, d_emb=d_emb, n_classes=n_classes,
                                       item_dim=item_dim, sq=sq, num_iter=num_iter)
    g = torch.Generator(device=device); g.manual_seed(999)
    items, queries, targets = make_batch(64, 16, n_classes, device, g)

    # Forward
    out_jack = model_jack(items, queries)
    out_tri = model_tri(items, queries)
    fwd_err = (out_jack - out_tri).abs().max().item()
    print(f"  forward logits max|Jack - Triton| = {fwd_err:.3e}")

    # Backward — compare grads on q_proj/k_proj/v_proj/phi
    loss_fn = nn.CrossEntropyLoss()
    model_jack.zero_grad(); model_tri.zero_grad()
    loss_fn(model_jack(items, queries), targets).backward()
    loss_fn(model_tri(items, queries), targets).backward()
    print("  parameter-gradient max|Jack - Triton|:")
    worst = 0.0
    for (n1, p1), (n2, p2) in zip(model_jack.named_parameters(),
                                  model_tri.named_parameters()):
        if p1.grad is None or p2.grad is None:
            continue
        e = (p1.grad - p2.grad).abs().max().item()
        denom = p1.grad.abs().max().item() + 1e-12
        worst = max(worst, e / denom)
        print(f"    {n1:18s}: abs={e:.3e}  rel={e/denom:.3e}")
    print(f"  >>> worst relative grad error = {worst:.3e}")
    print()
    return fwd_err, worst


def stage3_training(device, *, d_emb, n_classes, item_dim, sq, num_iter,
                    steps, id_len, ood_lens):
    print("=" * 78)
    print(f"STAGE 3 — full training comparison (q={sq}, {steps} steps)")
    print("=" * 78)
    results = {}
    for name, builder in [("JACK-bisection", "jack"), ("TRITON-BSmode", "tri")]:
        model_jack, model_tri = build_pair(device, d_emb=d_emb, n_classes=n_classes,
                                           item_dim=item_dim, sq=sq, num_iter=num_iter)
        model = model_jack if builder == "jack" else model_tri
        train(model, seq_len=id_len, n_classes=n_classes, device=device,
              steps=steps, bs=256, lr=1e-3, wd=1e-4, warmup=max(1, steps // 10), seed=0)
        accs = {}
        for L in [id_len] + ood_lens:
            accs[L] = evaluate(model, seq_len=L, n_classes=n_classes, device=device,
                               samples=2048 if L == id_len else 1024, bs=256)
        results[name] = accs
        print(f"  {name}: " + "  ".join(f"L{L}={a:.1f}" for L, a in accs.items()))
    print(f"\n  {'L':>6} | {'JACK':>8} | {'TRITON':>8} | {'Δ':>7}")
    print("  " + "-" * 36)
    for L in [id_len] + ood_lens:
        a = results["JACK-bisection"][L]; b = results["TRITON-BSmode"][L]
        print(f"  {L:>6} | {a:>7.2f}% | {b:>7.2f}% | {b-a:>+6.2f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=4.0)
    ap.add_argument("--num-iter", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--id-len", type=int, default=16)
    ap.add_argument("--ood-lens", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    n_classes = 10
    item_dim = 1 + n_classes
    d_emb = 128

    stage0_bug2_recheck(device)
    fwd_err, grad_err = stage1_2_fwd_grad(device, d_emb=d_emb, n_classes=n_classes,
                                          item_dim=item_dim, sq=args.q, num_iter=args.num_iter)
    train_res = stage3_training(device, d_emb=d_emb, n_classes=n_classes,
                                item_dim=item_dim, sq=args.q, num_iter=args.num_iter,
                                steps=args.steps, id_len=args.id_len, ood_lens=args.ood_lens)

    out = {
        "q": args.q, "num_iter": args.num_iter, "steps": args.steps,
        "forward_logit_max_err": fwd_err, "worst_rel_grad_err": grad_err,
        "training": {k: {str(L): v for L, v in d.items()} for k, d in train_res.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
