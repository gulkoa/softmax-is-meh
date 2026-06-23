"""
STEP 1: flash-FORWARD + torch-BACKWARD in architecture_new, vs the stieltjes
reference. Isolates the forward path (Triton flash forward) while using PyTorch
autograd for the backward.

Attention backend (_FlashAttnSingleQuery):
  - forward: tile the single query to Nq=Nk=T, run the Triton flash forward to
    get O (and lambda), take row 0, and RENORMALIZE by the weight sum (the new
    `stieltjes` reference does probs/probs.sum(); the flash kernel does not).
  - backward: recompute normalized weights in PyTorch with lambda DETACHED and
    use torch.autograd.grad. This matches the reference's gradient structure
    (bisection also detaches lambda via torch.where).

Compared, in architecture_new's MLP MaxRetrievalModel, against the normalized
`stieltjes` mapping with IDENTICAL weights / data / seed. Stages:
  (1) forward-logit equivalence, (2) parameter-gradient equivalence,
  (3) training accuracy at ID + OOD lengths.
Run fp32 (Jack's model is fp32). q in {2,4} (forward validated for q<=32).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import triton
from tqdm import tqdm

WORKTREE = "/users/PAS2402/alexg/softmax/psm-architecture-new"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")

from max_retrieval_architecture.architecture import MaxRetrievalModel  # noqa: E402
from mappings.type_enum import SimplexMappingEnum  # noqa: E402
import stieltjes_flash_attn as sfa  # noqa: E402


def _flash_fwd(q4, k4, v4, sm_scale, sq, num_iter):
    """Raw Triton forward; returns (O (B,H,N,D), lam (B*H, N) absolute)."""
    B, H, N, D = q4.shape
    o = torch.empty_like(q4)
    lam = torch.empty((B * H, N), device=q4.device, dtype=torch.float32)
    d_sum = torch.empty((B * H, N), device=q4.device, dtype=torch.float32)
    argmax = torch.empty((B * H, N), device=q4.device, dtype=torch.int32)
    lambda_init = torch.full((N,), 1.1, device=q4.device, dtype=torch.float32)
    if D <= 64:
        BM, BN = 128, 64
    elif q4.element_size() >= 4:
        BM, BN = 32, 32
    else:
        BM, BN = 64, 64
    grid = (triton.cdiv(N, BM), B * H)
    sfa._stieltjes_attn_fwd[grid](
        q4, k4, v4, o, lam, d_sum, argmax, lambda_init,
        q4.stride(0), q4.stride(1), q4.stride(2), q4.stride(3),
        k4.stride(0), k4.stride(1), k4.stride(2), k4.stride(3),
        v4.stride(0), v4.stride(1), v4.stride(2), v4.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale, N, sq=sq, NUM_ITER=num_iter, EPS=1e-6,
        HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, CAUSAL=False,
    )
    return o, lam


class _FlashAttnSingleQuery(torch.autograd.Function):
    """Single-query Stieltjes attention: flash forward + torch backward."""
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, sq, num_iter):
        # q: (B,1,d); k,v: (B,T,d)
        B, T, d = k.shape
        q4 = q.expand(B, T, d).unsqueeze(1).contiguous()  # (B,1,T,d)
        k4 = k.unsqueeze(1).contiguous()
        v4 = v.unsqueeze(1).contiguous()
        o, lam = _flash_fwd(q4, k4, v4, sm_scale, sq, num_iter)  # o (B,1,T,d), lam (B,T)
        z_unnorm = o[:, 0, 0, :]  # (B,d) — all query rows identical
        # weight sum S for query row 0 (renormalize, like the reference).
        # lam is the kernel's ABSOLUTE lambda; the kernel weights are
        # (lam - scores)^(-q) on RAW (uncentered) scores. Use raw scores here.
        scores = (q.float() @ k.float().transpose(-2, -1)).squeeze(1) * sm_scale  # (B,T)
        lam0 = lam[:, 0:1]                                  # (B,1)
        w = (lam0 - scores).clamp(min=1e-6).pow(-sq)        # (B,T)  raw scores
        S = w.sum(-1, keepdim=True).clamp(min=1e-9)         # (B,1)
        z = z_unnorm / S
        ctx.save_for_backward(q, k, v, lam0)
        ctx.sm_scale = sm_scale
        ctx.sq = sq
        return z

    @staticmethod
    def backward(ctx, dz):
        q, k, v, lam0 = ctx.saved_tensors
        sq, sm_scale = ctx.sq, ctx.sm_scale
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            scores = (qd @ kd.transpose(-2, -1)) * sm_scale     # (B,1,T)
            # Mirror the reference EXACTLY: center by a DIFFERENTIABLE max (this
            # is the argmax-gradient term), and use the kernel's CENTERED lambda
            # (lam0 - x_max), detached. Value is (lam0 - scores)^(-q); gradient
            # carries the argmax term, matching the bisection reference.
            x_max = scores.max(-1, keepdim=True).values         # differentiable
            shifted = scores - x_max
            lam_c = (lam0.detach().unsqueeze(1) - x_max).detach()  # centered, detached
            w = (lam_c - shifted).clamp(min=1e-6).pow(-sq)      # (B,1,T)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
            z = (w @ vd).squeeze(1)                              # (B,d)
            gq, gk, gv = torch.autograd.grad(z, [qd, kd, vd], dz)
        return gq, gk, gv, None, None, None


class MaxRetrievalModelFlash(MaxRetrievalModel):
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
        sm_scale = self.d_emb ** -0.5 if self.attn_score_scale == "inv_sqrt_d" else 1.0
        z = _FlashAttnSingleQuery.apply(q, k, v, sm_scale, self._sq, self._ni)  # (B,d)
        return self.phi(z)


# --- data / train / eval (architecture_new conventions) ---
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


def build_pair(device, *, d_emb, n_classes, item_dim, sq, num_iter):
    _set_seeds(0)
    ref = MaxRetrievalModel(simplex_mapping=SimplexMappingEnum.stieltjes, d_emb=d_emb,
                            n_classes=n_classes, item_input_dim=item_dim,
                            query_input_dim=1, attn_score_scale="inv_sqrt_d", q=sq).to(device)
    _set_seeds(0)
    fl = MaxRetrievalModelFlash(simplex_mapping=SimplexMappingEnum.stieltjes, d_emb=d_emb,
                                n_classes=n_classes, item_input_dim=item_dim,
                                query_input_dim=1, attn_score_scale="inv_sqrt_d", q=sq,
                                stieltjes_q=sq, num_iter=num_iter).to(device)
    fl.load_state_dict(ref.state_dict(), strict=False)
    return ref, fl


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

    # Stage 1+2: forward & gradient equivalence
    ref, fl = build_pair(device, d_emb=d_emb, n_classes=n_classes, item_dim=item_dim,
                         sq=args.q, num_iter=args.num_iter)
    g = torch.Generator(device=device); g.manual_seed(777)
    items, q, tgt = make_batch(64, 16, n_classes, device, g)
    out_ref, out_fl = ref(items, q), fl(items, q)
    fwd_err = (out_ref - out_fl).abs().max().item()
    print(f"STAGE 1 — forward logits max|ref - flash| = {fwd_err:.3e}")
    lf = nn.CrossEntropyLoss()
    ref.zero_grad(); fl.zero_grad()
    lf(ref(items, q), tgt).backward(); lf(fl(items, q), tgt).backward()
    worst = 0.0
    print("STAGE 2 — parameter-gradient max|ref - flash|:")
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), fl.named_parameters()):
        if p1.grad is None or p2.grad is None: continue
        e = (p1.grad - p2.grad).abs().max().item()
        rel = e / (p1.grad.abs().max().item() + 1e-12)
        worst = max(worst, rel)
        print(f"    {n1:14s}: abs={e:.3e} rel={rel:.3e}")
    print(f"  >>> worst rel grad err = {worst:.3e}\n")

    # Stage 3: training comparison
    print(f"STAGE 3 — training ({args.steps} steps)")
    results = {}
    for name, which in [("REF-stieltjes", "ref"), ("FLASH-fwd+torch-bwd", "fl")]:
        ref2, fl2 = build_pair(device, d_emb=d_emb, n_classes=n_classes, item_dim=item_dim,
                              sq=args.q, num_iter=args.num_iter)
        model = ref2 if which == "ref" else fl2
        train(model, seq_len=args.id_len, n_classes=n_classes, device=device,
              steps=args.steps, bs=256, lr=1e-3, wd=1e-4, warmup=max(1, args.steps // 10), seed=0)
        accs = {L: evaluate(model, seq_len=L, n_classes=n_classes, device=device,
                            samples=2048 if L == args.id_len else 1024, bs=256)
                for L in [args.id_len] + args.ood_lens}
        results[name] = accs
        print(f"  {name}: " + "  ".join(f"L{L}={a:.1f}" for L, a in accs.items()))
    print(f"\n  {'L':>6} | {'REF':>8} | {'FLASH':>8} | {'Δ':>7}")
    for L in [args.id_len] + args.ood_lens:
        a, b = results["REF-stieltjes"][L], results["FLASH-fwd+torch-bwd"][L]
        print(f"  {L:>6} | {a:>7.2f}% | {b:>7.2f}% | {b-a:>+6.2f}")

    out = {"q": args.q, "num_iter": args.num_iter, "steps": args.steps,
           "fwd_err": fwd_err, "worst_rel_grad_err": worst,
           "training": {k: {str(L): v for L, v in d.items()} for k, d in results.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
