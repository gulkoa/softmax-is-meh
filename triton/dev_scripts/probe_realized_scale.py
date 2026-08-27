"""G2 gate probe (plan 2026-08-16): realized attention-score scale per
head across contexts, comparing pilot ckpts. For each ckpt and ctx in
{512,1024,2048}: run a few val batches, capture qkv outputs via forward
hook, replay the module's own q/k transforms (qk-norm gains, rope,
capped/uncapped scale mult), and record p50/p99 of |scores| after
1/sqrt(hd). The 07-18 exploder signature was score-scale p99 ~10x the
healthy band; gate G2 requires B within 2x of A.

Usage: python probe_realized_scale.py ckptA.pt ckptB.pt ... \
           [--data-mix same-as-training] [--batches 4] [--out out.json]
"""
import argparse, json, math, sys

import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts")
from types import SimpleNamespace
from train_gpt2_stieltjes import GPT, Shards

DEVICE = torch.device("cuda")


def probe(ckpt_path, ctxs, batches, mix):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    model.eval()
    attns = [m for m in model.modules() if hasattr(m, "qkv")]
    cap = {}
    hooks = [a.qkv.register_forward_hook(
        (lambda mod, i, o, a=a: cap.__setitem__(a, o))) for a in attns]
    import numpy as np
    val = Shards("val", mix)
    rng = np.random.default_rng(0)
    out = {}
    for ctx in ctxs:
        stats = []
        for b in range(batches):
            ids, _ = val.batch(2, ctx, rng, DEVICE)
            with torch.no_grad():
                model(ids)
            for a in attns:
                o = cap[a]
                B, S, E3 = o.shape
                E = E3 // 3
                q, k, _ = o.split(E, dim=2)
                q = q.view(B, S, a.h, a.hd).transpose(1, 2)
                k = k.view(B, S, a.h, a.hd).transpose(1, 2)
                if getattr(a, "qk_norm", False):
                    q = a._rms_head(q, a.q_gain)
                    k = a._rms_head(k, a.k_gain)
                if getattr(a, "rope", False):
                    cos, sin = a._rope_cos_sin(S, q.device)
                    q = a._apply_rope(q, cos, sin)
                    k = a._apply_rope(k, cos, sin)
                if hasattr(a, "scale_mult"):
                    C = float(getattr(a, "scale_cap", 15.0))
                    eff = (1.0 + C * torch.tanh(a.scale_mult / C)
                           if C > 0 else 1.0 + a.scale_mult)
                    q = q * eff[None, :, None, None].to(q.dtype)
                s = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float())
                s = (s / math.sqrt(a.hd)).abs().flatten()
                # subsample for quantiles: exact quantile on the full
                # score tensor OOMs at ctx 2048 x 12 layers
                idx = torch.randint(s.numel(), (min(2_000_000, s.numel()),),
                                    device=s.device)
                sub = s[idx]
                stats.append(torch.quantile(
                    sub, torch.tensor([0.5, 0.99], device=s.device)).cpu())
                del s, sub, cap[a]
                torch.cuda.empty_cache()
        st = torch.stack(stats)          # (n, 2, B?, H) -> aggregate
        out[ctx] = {"p50": float(st[:, 0].median()),
                    "p99_mean": float(st[:, 1].mean()),
                    "p99_max": float(st[:, 1].max())}
        print(f"  ctx {ctx}: p50 {out[ctx]['p50']:.2f} "
              f"p99mean {out[ctx]['p99_mean']:.2f} "
              f"p99max {out[ctx]['p99_max']:.2f}", flush=True)
    for h in hooks:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data-mix", default=None, dest="data_mix")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--out", default="results/qknorm_realized_scale.json")
    a = ap.parse_args()
    res = {}
    for c in a.ckpts:
        print(f"== {c}", flush=True)
        res[c] = probe(c, (512, 1024, 2048), a.batches, a.data_mix)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
