"""
STEP 3: full Triton flash kernel (forward + BACKWARD, block_lambda_grad=True)
swapped into architecture_new, vs the PyTorch reference.

Now that the forward is validated (flashfwd_torchbwd_archnew.py), test the flash
BACKWARD (the custom Triton 3-kernel backward). The kernel computes the
UN-normalized Stieltjes weights (lam - s)^(-q) and the BS-style gradient
(detached lambda + argmax correction), so its native PyTorch match is
`stieltjes_old` (no probs/probs.sum()). We compare against:
  - stieltjes_old (unnormalized)  -> primary, should match to fp32 precision
  - stieltjes (normalized)        -> off by the per-row weight-sum factor
                                     (~1e-4 at q<=4 where sum~=1.0001)

fp32, D=128 -> exercises the precision-aware block-size fix (32x32) that
resolves the fp32 backward shared-memory OOM. q in {2,4}.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

WORKTREE = "/users/PAS2402/alexg/softmax/psm-architecture-new"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")

from max_retrieval_architecture.architecture import MaxRetrievalModel  # noqa: E402
from mappings.type_enum import SimplexMappingEnum  # noqa: E402
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402


class MaxRetrievalModelFlashFull(MaxRetrievalModel):
    """architecture_new MLP model with attention via the FULL Triton kernel
    (flash forward + flash backward, block_lambda_grad=True). Unnormalized."""
    def __init__(self, *a, stieltjes_q=4.0, num_iter=15, **kw):
        super().__init__(*a, **kw)
        self._sq = float(stieltjes_q)
        self._ni = int(num_iter)

    def forward(self, x_items, x_query, return_attn=False):
        h_items = self.psi_x(x_items)
        h_query = self.psi_q(x_query.unsqueeze(-1))
        q = self.q_proj(h_query)             # (B,1,d)
        k = self.k_proj(h_items)             # (B,T,d)
        v = self.v_proj(h_items)             # (B,T,d)
        B, T, d = k.shape
        sm = self.d_emb ** -0.5 if self.attn_score_scale == "inv_sqrt_d" else 1.0
        q4 = q.expand(B, T, d).unsqueeze(1).contiguous()  # (B,1,T,d)
        k4 = k.unsqueeze(1).contiguous()
        v4 = v.unsqueeze(1).contiguous()
        o = stieltjes_attention(q4, k4, v4, causal=False, sm_scale=sm,
                                stieltjes_q=self._sq, num_iter=self._ni,
                                block_lambda_grad=True)   # (B,1,T,d) UNNORMALIZED
        z = o[:, 0, 0, :]                                 # (B,d)
        return self.phi(z)


def _set_seeds(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def make_batch(bs, T, n_classes, device, gen):
    pri = torch.rand(bs, T, device=device, generator=gen)
    cls = torch.randint(0, n_classes, (bs, T), device=device, generator=gen)
    tgt = cls.gather(1, pri.argmax(1, keepdim=True)).squeeze(1).long()
    items = torch.cat([pri.unsqueeze(-1), F.one_hot(cls, n_classes).float()], -1)
    q = torch.rand(bs, 1, device=device, generator=gen)
    return items, q, tgt


def train(model, *, seq_len, n_classes, device, steps, bs, lr, wd, warmup, seed):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                        pct_start=warmup / steps, anneal_strategy="cos")
    lf = nn.CrossEntropyLoss()
    for step in tqdm(range(steps), desc="train", leave=False):
        g = torch.Generator(device=device); g.manual_seed(seed * 1_000_003 + step)
        items, q, tgt = make_batch(bs, seq_len, n_classes, device, g)
        opt.zero_grad(set_to_none=True)
        loss = lf(model(items, q), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()


@torch.no_grad()
def evaluate(model, *, seq_len, n_classes, device, samples, bs, seed=12345):
    model.eval(); correct = total = 0; ebs = bs * 2
    for i in range(int(np.ceil(samples / ebs))):
        g = torch.Generator(device=device); g.manual_seed(seed + i)
        items, q, tgt = make_batch(ebs, seq_len, n_classes, device, g)
        correct += (model(items, q).argmax(-1) == tgt).sum().item(); total += tgt.numel()
    return 100.0 * correct / max(total, 1)


def build_ref(kind, device, *, d_emb, n_classes, item_dim, sq):
    _set_seeds(0)
    mp = (SimplexMappingEnum.stieltjes_old if kind == "old"
          else SimplexMappingEnum.stieltjes)
    return MaxRetrievalModel(simplex_mapping=mp, d_emb=d_emb, n_classes=n_classes,
                             item_input_dim=item_dim, query_input_dim=1,
                             attn_score_scale="inv_sqrt_d", q=sq).to(device)


def build_flash(device, *, d_emb, n_classes, item_dim, sq, num_iter):
    _set_seeds(0)
    return MaxRetrievalModelFlashFull(
        simplex_mapping=SimplexMappingEnum.stieltjes, d_emb=d_emb, n_classes=n_classes,
        item_input_dim=item_dim, query_input_dim=1, attn_score_scale="inv_sqrt_d",
        q=sq, stieltjes_q=sq, num_iter=num_iter).to(device)


def grad_compare(ref, flash, items, q, tgt):
    lf = nn.CrossEntropyLoss()
    ref.zero_grad(); flash.zero_grad()
    lf(ref(items, q), tgt).backward()
    lf(flash(items, q), tgt).backward()
    worst = 0.0
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), flash.named_parameters()):
        if p1.grad is None or p2.grad is None: continue
        e = (p1.grad - p2.grad).abs().max().item()
        denom = p1.grad.abs().max().item() + 1e-12
        if denom > 1e-8:  # skip near-zero gradients (meaningless rel)
            worst = max(worst, e / denom)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=4.0)
    ap.add_argument("--num-iter", type=int, default=15)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--id-len", type=int, default=16)
    ap.add_argument("--ood-lens", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  q={args.q}  steps={args.steps}\n")
    n_classes, d_emb = 10, 128
    item_dim = 1 + n_classes

    # Stage 1+2: forward & gradient equivalence vs both references
    flash = build_flash(device, d_emb=d_emb, n_classes=n_classes, item_dim=item_dim,
                        sq=args.q, num_iter=args.num_iter)
    g = torch.Generator(device=device); g.manual_seed(777)
    items, q, tgt = make_batch(64, 16, n_classes, device, g)
    print("STAGE 1+2 — flash (full kernel) vs references:")
    for kind in ["old", "new"]:
        ref = build_ref(kind, device, d_emb=d_emb, n_classes=n_classes,
                       item_dim=item_dim, sq=args.q)
        flash.load_state_dict(ref.state_dict(), strict=False)
        fwd = (ref(items, q) - flash(items, q)).abs().max().item()
        gerr = grad_compare(ref, flash, items, q, tgt)
        label = "stieltjes_old(unnorm)" if kind == "old" else "stieltjes(norm)"
        print(f"  vs {label:22s}: forward max|Δ|={fwd:.3e}  worst_rel_grad={gerr:.3e}")
    print()

    # Stage 3: training, flash vs stieltjes_old (native match)
    print(f"STAGE 3 — training ({args.steps} steps)")
    results = {}
    for name, builder in [("REF-stieltjes_old", "old"),
                          ("REF-stieltjes", "new"),
                          ("FLASH-full(fwd+bwd)", "flash")]:
        if builder == "flash":
            model = build_flash(device, d_emb=d_emb, n_classes=n_classes,
                               item_dim=item_dim, sq=args.q, num_iter=args.num_iter)
        else:
            model = build_ref(builder, device, d_emb=d_emb, n_classes=n_classes,
                             item_dim=item_dim, sq=args.q)
        train(model, seq_len=args.id_len, n_classes=n_classes, device=device,
              steps=args.steps, bs=256, lr=1e-3, wd=1e-4,
              warmup=max(1, args.steps // 10), seed=0)
        accs = {L: evaluate(model, seq_len=L, n_classes=n_classes, device=device,
                            samples=2048 if L == args.id_len else 1024, bs=256)
                for L in [args.id_len] + args.ood_lens}
        results[name] = accs
        print(f"  {name:22s}: " + "  ".join(f"L{L}={a:.1f}" for L, a in accs.items()))

    print(f"\n  {'L':>6} | {'stj_old':>8} | {'stj':>8} | {'FLASH':>8} | {'F-old':>7}")
    for L in [args.id_len] + args.ood_lens:
        a = results["REF-stieltjes_old"][L]
        b = results["REF-stieltjes"][L]
        c = results["FLASH-full(fwd+bwd)"][L]
        print(f"  {L:>6} | {a:>7.2f}% | {b:>7.2f}% | {c:>7.2f}% | {c-a:>+6.2f}")

    out = {"q": args.q, "num_iter": args.num_iter, "steps": args.steps,
           "training": {k: {str(L): v for L, v in d.items()} for k, d in results.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
