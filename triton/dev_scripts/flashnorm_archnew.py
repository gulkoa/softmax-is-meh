"""
END-TO-END: full Triton kernel in NORMALIZED mode (normalize=True, flash fwd +
flash bwd) inside architecture_new, vs the normalized `stieltjes` PyTorch
reference. The kernel must reproduce the reference's superior OOD accuracy.

Stages: (1) forward-logit equivalence, (2) parameter-gradient equivalence,
(3) full training comparison (REF-stieltjes vs FLASH-norm; stieltjes_old shown
for context — the normalized variants should both beat it OOD).
fp32, q ∈ given list.
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


class MaxRetrievalModelFlashNorm(MaxRetrievalModel):
    """architecture_new MLP model; attention via the full Triton kernel in
    NORMALIZED mode (flash forward + flash backward)."""
    def __init__(self, *a, stieltjes_q=4.0, num_iter=15, **kw):
        super().__init__(*a, **kw)
        self._sq = float(stieltjes_q)
        self._ni = int(num_iter)

    def forward(self, x_items, x_query, return_attn=False):
        h_items = self.psi_x(x_items)
        h_query = self.psi_q(x_query.unsqueeze(-1))
        q = self.q_proj(h_query)
        k = self.k_proj(h_items)
        v = self.v_proj(h_items)
        B, T, d = k.shape
        sm = self.d_emb ** -0.5 if self.attn_score_scale == "inv_sqrt_d" else 1.0
        q4 = q.expand(B, T, d).unsqueeze(1).contiguous()
        k4 = k.unsqueeze(1).contiguous()
        v4 = v.unsqueeze(1).contiguous()
        o = stieltjes_attention(q4, k4, v4, causal=False, sm_scale=sm,
                                stieltjes_q=self._sq, num_iter=self._ni,
                                normalize=True)
        z = o[:, 0, 0, :]
        return self.phi(z)


def _set_seeds(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def make_batch(bs, T, n_classes, device, gen):
    pri = torch.rand(bs, T, device=device, generator=gen)
    cls = torch.randint(0, n_classes, (bs, T), device=device, generator=gen)
    tgt = cls.gather(1, pri.argmax(1, keepdim=True)).squeeze(1).long()
    items = torch.cat([pri.unsqueeze(-1), F.one_hot(cls, n_classes).float()], -1)
    qv = torch.rand(bs, 1, device=device, generator=gen)
    return items, qv, tgt


def train(model, *, seq_len, n_classes, device, steps, bs, lr, wd, warmup, seed):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                        pct_start=warmup / steps,
                                        anneal_strategy="cos")
    lf = nn.CrossEntropyLoss()
    for step in tqdm(range(steps), desc="train", leave=False):
        g = torch.Generator(device=device); g.manual_seed(seed * 1_000_003 + step)
        items, qv, tgt = make_batch(bs, seq_len, n_classes, device, g)
        opt.zero_grad(set_to_none=True)
        loss = lf(model(items, qv), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()


@torch.no_grad()
def evaluate(model, *, seq_len, n_classes, device, samples, bs, seed=12345):
    model.eval(); correct = total = 0; ebs = bs * 2
    for i in range(int(np.ceil(samples / ebs))):
        g = torch.Generator(device=device); g.manual_seed(seed + i)
        items, qv, tgt = make_batch(ebs, seq_len, n_classes, device, g)
        correct += (model(items, qv).argmax(-1) == tgt).sum().item()
        total += tgt.numel()
    return 100.0 * correct / max(total, 1)


def build_ref(mapping_enum, device, *, d_emb, n_classes, item_dim, sq, seed):
    _set_seeds(seed)
    return MaxRetrievalModel(simplex_mapping=mapping_enum, d_emb=d_emb,
                             n_classes=n_classes, item_input_dim=item_dim,
                             query_input_dim=1, attn_score_scale="inv_sqrt_d",
                             q=sq).to(device)


def build_flash(device, *, d_emb, n_classes, item_dim, sq, num_iter, seed):
    _set_seeds(seed)
    return MaxRetrievalModelFlashNorm(
        simplex_mapping=SimplexMappingEnum.stieltjes, d_emb=d_emb,
        n_classes=n_classes, item_input_dim=item_dim, query_input_dim=1,
        attn_score_scale="inv_sqrt_d", q=sq,
        stieltjes_q=sq, num_iter=num_iter).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--num-iter", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--id-len", type=int, default=16)
    ap.add_argument("--ood-lens", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  steps={args.steps}\n")
    n_classes, d_emb = 10, 128
    item_dim = 1 + n_classes
    lengths = [args.id_len] + args.ood_lens
    lf = nn.CrossEntropyLoss()

    all_out = {}
    for sq in args.qs:
        print("=" * 78)
        print(f"q = {sq}")
        print("=" * 78)

        # Stage 1+2: forward & gradient equivalence vs normalized reference
        ref = build_ref(SimplexMappingEnum.stieltjes, device, d_emb=d_emb,
                        n_classes=n_classes, item_dim=item_dim, sq=sq, seed=0)
        fl = build_flash(device, d_emb=d_emb, n_classes=n_classes,
                         item_dim=item_dim, sq=sq, num_iter=args.num_iter, seed=0)
        fl.load_state_dict(ref.state_dict(), strict=False)
        g = torch.Generator(device=device); g.manual_seed(777)
        items, qv, tgt = make_batch(64, args.id_len, n_classes, device, g)
        fwd_err = (ref(items, qv) - fl(items, qv)).abs().max().item()
        print(f"STAGE 1 — forward logits max|ref-flash| = {fwd_err:.3e}")
        ref.zero_grad(); fl.zero_grad()
        lf(ref(items, qv), tgt).backward()
        lf(fl(items, qv), tgt).backward()
        worst = 0.0
        for (n1, p1), (_n2, p2) in zip(ref.named_parameters(), fl.named_parameters()):
            if p1.grad is None or p2.grad is None:
                continue
            denom = p1.grad.abs().max().item()
            if denom > 1e-8:
                worst = max(worst, (p1.grad - p2.grad).abs().max().item() / denom)
        print(f"STAGE 2 — worst rel param-grad err = {worst:.3e}")

        # Stage 3: training
        print(f"STAGE 3 — training ({args.steps} steps)")
        results = {}
        runs = [("REF-stieltjes_old", "old"), ("REF-stieltjes", "norm"),
                ("FLASH-normalized", "flash")]
        for name, kind in runs:
            if kind == "flash":
                model = build_flash(device, d_emb=d_emb, n_classes=n_classes,
                                    item_dim=item_dim, sq=sq,
                                    num_iter=args.num_iter, seed=0)
            else:
                enum = (SimplexMappingEnum.stieltjes_old if kind == "old"
                        else SimplexMappingEnum.stieltjes)
                model = build_ref(enum, device, d_emb=d_emb, n_classes=n_classes,
                                  item_dim=item_dim, sq=sq, seed=0)
            train(model, seq_len=args.id_len, n_classes=n_classes, device=device,
                  steps=args.steps, bs=256, lr=1e-3, wd=1e-4,
                  warmup=max(1, args.steps // 10), seed=0)
            accs = {L: evaluate(model, seq_len=L, n_classes=n_classes,
                                device=device,
                                samples=2048 if L == args.id_len else 1024, bs=256)
                    for L in lengths}
            results[name] = accs
            print(f"  {name:18s}: " + "  ".join(f"L{L}={a:.1f}"
                                                for L, a in accs.items()))
        print(f"\n  {'L':>6} | {'stj_old':>8} | {'stj':>8} | {'FLASHn':>8} | {'Fn-stj':>7}")
        for L in lengths:
            a = results["REF-stieltjes_old"][L]
            b = results["REF-stieltjes"][L]
            c = results["FLASH-normalized"][L]
            print(f"  {L:>6} | {a:>7.2f}% | {b:>7.2f}% | {c:>7.2f}% | {c-b:>+6.2f}")
        all_out[str(sq)] = {
            "fwd_err": fwd_err, "worst_rel_grad": worst,
            "training": {k: {str(L): v for L, v in d.items()}
                         for k, d in results.items()},
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
