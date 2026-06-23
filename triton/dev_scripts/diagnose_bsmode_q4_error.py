"""
Diagnose the q=4 BS-mode dQ/dK 25% error (separate from the NaN bug).

Hypothesis: it's ARGMAX FRAGILITY, not a logic error.
  The BS-mode backward subtracts kappa_i (a large per-row scalar) at the single
  column j == argmax_i. If Triton's fp16 argmax disagrees with PyTorch's argmax
  on a near-tied row, the large kappa lands on the WRONG column -> big local error
  in dS -> big max|dQ|, max|dK|. Most elements are fine; the MAX is dominated by
  the handful of mismatched rows.

Three falsifiable tests:
  (A) ARGMAX CORRECTNESS: call the forward kernel directly, read the stored
      argmax, compare to torch.argmax(scores) per row. Count mismatches at
      fp16 vs fp32. Predict: fp16 has some mismatches on near-ties; fp32 ~0.
  (B) fp32 BS-mode gradient: rerun the gradient check with fp32 inputs.
      Predict: error collapses to ~1e-5 (logic is correct; fp16 was the issue).
  (C) LOCALIZATION: for fp16, show that the max-error rows are exactly the
      argmax-mismatch rows, and report the top-2 score gap on those rows
      (near-ties).
"""
import sys
import torch
import triton

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")

import stieltjes_flash_attn as sfa
from stieltjes_flash_attn import stieltjes_attention
from maxretr_bs_vs_nr import StieltjesBSTransform


def run_forward_kernel_get_argmax(q, k, v, sm_scale, stieltjes_q, num_iter):
    """Call _stieltjes_attn_fwd directly and return the stored argmax tensor
    plus lambda, so we can inspect them. Mirrors StieltjesAttention.forward."""
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    lam = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    d_sum = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    argmax = torch.empty((B * H, N), device=q.device, dtype=torch.int32)
    lambda_init = torch.full((N,), 1.1, device=q.device, dtype=torch.float32)

    if D <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    sfa._stieltjes_attn_fwd[grid](
        q, k, v, o, lam, d_sum, argmax, lambda_init,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale, N,
        sq=stieltjes_q, NUM_ITER=num_iter, EPS=1e-6,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=False,
    )
    return argmax, lam, o


def test_A_argmax_correctness(device):
    print("=" * 78)
    print("(A) ARGMAX CORRECTNESS: Triton-stored argmax vs torch.argmax(scores)")
    print("=" * 78)
    for N in [64, 128]:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            torch.manual_seed(0)
            B, H, D = 2, 4, 64
            sm_scale = 1.0 / (D ** 0.5)
            q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            k = torch.randn(B, H, N, D, device=device, dtype=dtype)
            v = torch.randn(B, H, N, D, device=device, dtype=dtype)

            argmax_tri, lam, o = run_forward_kernel_get_argmax(
                q, k, v, sm_scale, 4.0, 10)
            argmax_tri = argmax_tri.view(B, H, N)  # (BH, N) -> (B,H,N)

            # Reference scores computed the SAME way the kernel does internally:
            # qk = (q @ k^T) * scale, in fp32 after the matmul
            scores = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
            argmax_ref = scores.argmax(dim=-1).to(torch.int32)  # (B,H,N)

            mism = (argmax_tri != argmax_ref)
            n_mismatch = mism.sum().item()
            total = argmax_ref.numel()
            print(f"  N={N:4d} {dtype_name}: {n_mismatch:4d}/{total} argmax mismatches "
                  f"({100.0*n_mismatch/total:.2f}%)")

            # For mismatched rows, report the top-2 score gap (near-tie indicator)
            if n_mismatch > 0:
                flat_scores = scores.reshape(-1, N)
                flat_mism = mism.reshape(-1)
                idx = torch.nonzero(flat_mism).flatten()[:5]
                for j in idx.tolist():
                    row = flat_scores[j]
                    top2 = torch.topk(row, 2).values
                    gap = (top2[0] - top2[1]).item()
                    ti = argmax_tri.reshape(-1)[j].item()
                    ri = argmax_ref.reshape(-1)[j].item()
                    print(f"      row {j}: triton_argmax={ti} ref_argmax={ri} "
                          f"top2_gap={gap:.3e} score[tri]={row[ti].item():.4f} "
                          f"score[ref]={row[ri].item():.4f}")
        print()


def test_B_fp32_bsmode(device):
    print("=" * 78)
    print("(B) BS-mode gradient error: fp16 vs fp32 inputs")
    print("    Predict: fp32 error ~1e-5 (logic correct), fp16 error large (argmax)")
    print("=" * 78)
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        for q_val in [1.0, 4.0, 16.0]:
            torch.manual_seed(42)
            B, H, N, D = 2, 4, 64, 64
            sm_scale = 1.0 / (D ** 0.5)
            qf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            kf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            vf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

            # PyTorch BS reference (fp32)
            bs_map = StieltjesBSTransform(q=q_val, num_iter=64, eps=1e-9)
            qb = qf.clone().requires_grad_(True)
            kb = kf.clone().requires_grad_(True)
            vb = vf.clone().requires_grad_(True)
            scores = torch.matmul(qb, kb.transpose(-2, -1)) * sm_scale
            w = bs_map.translate_logits(scores, dim=-1)
            o_bs = torch.matmul(w, vb)
            do = torch.randn_like(o_bs)
            o_bs.backward(do)
            dq_bs, dk_bs = qb.grad.clone(), kb.grad.clone()

            # Triton BS-mode at this dtype
            qt = qf.clone().to(dtype).requires_grad_(True)
            kt = kf.clone().to(dtype).requires_grad_(True)
            vt = vf.clone().to(dtype).requires_grad_(True)
            o_t = stieltjes_attention(qt, kt, vt, causal=False, sm_scale=sm_scale,
                                      stieltjes_q=q_val, num_iter=10,
                                      block_lambda_grad=True)
            o_t.backward(do.to(dtype))
            dq_t, dk_t = qt.grad.float(), kt.grad.float()

            dq_err = (dq_t - dq_bs).abs().max().item()
            dk_err = (dk_t - dk_bs).abs().max().item()
            dq_rel = dq_err / dq_bs.abs().max().item()
            dk_rel = dk_err / dk_bs.abs().max().item()
            print(f"  {dtype_name} q={q_val:4.0f}: dQ_err={dq_err:.3e} (rel {dq_rel:.3e})  "
                  f"dK_err={dk_err:.3e} (rel {dk_rel:.3e})")
        print()


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    test_A_argmax_correctness(device)
    test_B_fp32_bsmode(device)


if __name__ == "__main__":
    main()
