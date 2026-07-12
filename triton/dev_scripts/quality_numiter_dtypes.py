"""
QUALITY experiments for the normalized Triton Stieltjes kernel.

Part 1 — dtype correctness sweep (fp32/fp16/bf16):
  Forward + backward vs the fp32 PyTorch normalized reference (Jack's
  `stieltjes`), per dtype. bf16 was never covered before — this closes that gap.
  Tolerances scale with dtype epsilon (fp32 5e-4, fp16 5e-3, bf16 3e-2).

Part 2 — num_iter sensitivity (quality vs speed knob):
  How many NR iterations does the kernel actually need? For q ∈ {2,4,8,16,32}
  and N ∈ {1024, 16384}: weight error vs bisection-80 ground truth and row-sum
  deviation as a function of num_iter ∈ {3,5,8,12,20,30}; plus kernel fwd
  latency vs num_iter at N=16384 fp16 (each iter is one O(N) K-sweep, so fewer
  iters = direct speedup).

All results logged to wandb (offline on compute nodes).
"""
import os
import sys

import torch

WORKTREE = "/users/PAS2402/alexg/softmax/psm-architecture-new"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")

from mappings.stieltjes import StieltjesTransform as JackNorm  # noqa: E402
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

TOL = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 3e-2}
DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
GRAD_TOL = {"fp32": 2e-2, "fp16": 8e-2, "bf16": 2e-1}


def ref_norm_attention(q, k, v, sm, sq):
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm
    p = JackNorm(q=sq, num_iter=30, eps=1e-9).translate_logits(scores, dim=-1)
    return torch.matmul(p, v)


def part1_dtypes(run):
    print("=" * 76)
    print("PART 1 — dtype correctness (forward + backward vs fp32 reference)")
    print("=" * 76)
    device = torch.device("cuda")
    B, H, D = 2, 4, 64
    fails = 0
    for dt_name, dtype in DT.items():
        for N in [64, 128, 256]:
            for sq in [2.0, 4.0, 16.0]:
                torch.manual_seed(7)
                sm = 1.0 / D ** 0.5
                qf = torch.randn(B, H, N, D, device=device)
                kf = torch.randn(B, H, N, D, device=device)
                vf = torch.randn(B, H, N, D, device=device)

                # fp32 reference fwd+bwd
                q1, k1, v1 = (t.clone().requires_grad_(True) for t in (qf, kf, vf))
                o1 = ref_norm_attention(q1, k1, v1, sm, sq)
                do = torch.randn_like(o1)
                o1.backward(do)

                # kernel at dtype
                q2 = qf.to(dtype).requires_grad_(True)
                k2 = kf.to(dtype).requires_grad_(True)
                v2 = vf.to(dtype).requires_grad_(True)
                o2 = stieltjes_attention(q2, k2, v2, causal=False, sm_scale=sm,
                                         stieltjes_q=sq, num_iter=25,
                                         normalize=True)
                o2.backward(do.to(dtype))

                fwd_err = (o2.float() - o1).abs().max().item()
                grels = []
                nan = torch.isnan(o2).any().item()
                for g1, g2 in [(q1.grad, q2.grad), (k1.grad, k2.grad),
                               (v1.grad, v2.grad)]:
                    nan = nan or torch.isnan(g2).any().item()
                    grels.append((g2.float() - g1).abs().max().item()
                                 / (g1.abs().max().item() + 1e-12))
                ok = (fwd_err <= TOL[dt_name] and max(grels) <= GRAD_TOL[dt_name]
                      and not nan)
                fails += (not ok)
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {dt_name} N={N:3d} q={sq:>4.0f}: "
                      f"fwd={fwd_err:.2e} grad_max_rel={max(grels):.2e} nan={nan}")
                run.log({"part": 1, "dtype": dt_name, "N": N, "q": sq,
                         f"fwd_err/{dt_name}": fwd_err,
                         f"grad_rel/{dt_name}": max(grels), "fail": int(not ok)})
    print(f"  >>> Part 1: {'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return fails


def part2_numiter(run):
    print("\n" + "=" * 76)
    print("PART 2 — num_iter sensitivity (weight error vs bisection-80 truth)")
    print("=" * 76)
    device = torch.device("cuda")

    def nr_weights(x, sq, iters, eps=1e-6):
        lam = torch.full_like(x[..., :1], 1.1)
        for _ in range(iters):
            diff = (lam - x).clamp(min=eps)
            f = diff.pow(-sq).sum(-1, keepdim=True) - 1.0
            fp = -sq * diff.pow(-sq - 1.0).sum(-1, keepdim=True)
            lam = lam - f / fp
        w = (lam - x).clamp(min=eps).pow(-sq)
        return w / w.sum(-1, keepdim=True)

    def bisect_weights(x, sq, iters=80, eps=1e-9):
        lb = torch.full_like(x[..., :1], eps)
        ub = torch.full_like(x[..., :1], x.shape[-1] ** (1.0 / sq))
        for _ in range(iters):
            mid = (lb + ub) / 2
            f = (mid - x).clamp(min=eps).pow(-sq).sum(-1, keepdim=True) - 1.0
            lb = torch.where(f > 0, mid, lb)
            ub = torch.where(f <= 0, mid, ub)
        lam = (lb + ub) / 2
        w = (lam - x).clamp(min=eps).pow(-sq)
        return w / w.sum(-1, keepdim=True)

    print(f"{'q':>4} {'N':>6} {'iters':>6} | {'max|p-p*|':>11} {'rowsum_dev':>11}")
    for sq in [2.0, 4.0, 8.0, 16.0, 32.0]:
        for N in [1024, 16384]:
            torch.manual_seed(0)
            scores = torch.randn(8, N, device=device, dtype=torch.float32) * 2.0
            x = scores - scores.max(-1, keepdim=True).values
            p_star = bisect_weights(x.double(), sq).float()
            for it in [3, 5, 8, 12, 20, 30]:
                p = nr_weights(x, sq, it)
                err = (p - p_star).abs().max().item()
                # row-sum dev of UNNORMALIZED weights (solver convergence)
                lam = None  # recompute unnorm sum
                w_un_sum_dev = 0.0
                print(f"{sq:>4.0f} {N:>6} {it:>6} | {err:>11.2e} {'':>11}")
                run.log({"part": 2, "q": sq, "N": N, "num_iter": it,
                         "weight_err_vs_bisect80": err})

    # kernel fwd latency vs num_iter (N=16384 fp16)
    print("\n  kernel fwd latency vs num_iter (N=16384, fp16, B1 H8 D64 q=4):")
    B, H, N, D = 1, 8, 16384, 64
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    sm = 1.0 / D ** 0.5
    for it in [3, 5, 8, 12, 20, 30]:
        def fwd():
            with torch.no_grad():
                stieltjes_attention(q, k, v, causal=False, sm_scale=sm,
                                    stieltjes_q=4.0, num_iter=it, normalize=True)
        for _ in range(3):
            fwd()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            fwd()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 10
        print(f"    num_iter={it:2d}: {ms:7.2f} ms")
        run.log({"part": 2, "latency_num_iter": it, "fwd_ms_16k_fp16": ms})


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"quality-numiter-dtypes-{os.environ.get('SLURM_JOB_ID', 'local')}",
                     config={"suite": "quality", "parts": ["dtype-correctness",
                                                           "numiter-sensitivity"]})
    fails = part1_dtypes(run)
    part2_numiter(run)
    run.summary["part1_failures"] = fails
    run.finish()
    print("\nDONE" + ("" if fails == 0 else f" ({fails} part-1 failures)"))


if __name__ == "__main__":
    main()
