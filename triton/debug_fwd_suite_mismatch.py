"""Localize the stale in-file fwd suite mismatch (task #39).

Smallest failing config (B=1 H=1 N=64 D=64 causal=False q=1.0,
max_err 0.155): dump per-row lambda and weight-sum from BOTH the Triton
kernel (raw launch, branch signature with H) and the fp32 reference,
same fp16 inputs. Separates solver disagreement (lambda differs) from
weight/materialization disagreement (lambda matches, sum_w differs).

Run on GPU: uv run --project <triton venv> python debug_fwd_suite_mismatch.py
"""
import torch
import triton

import stieltjes_flash_attn as sfa

DEVICE = torch.device("cuda")


def main():
    torch.manual_seed(0)
    B, H, N, D, sq = 1, 1, 64, 64, 1.0
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    sm_scale = 1.0 / (D ** 0.5)

    # --- reference lambda / weights (fp32, exactly as the ref computes)
    scores = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    s_max = scores.max(dim=-1, keepdim=True).values
    x = scores - s_max
    lam_ref = torch.full_like(s_max, 1.1)
    for _ in range(10):
        diff = (lam_ref - x).clamp(min=1e-6)
        f = diff.pow(-sq).sum(-1, keepdim=True) - 1.0
        fd = -sq * diff.pow(-sq - 1.0).sum(-1, keepdim=True)
        lam_ref = lam_ref - f / fd
    w_ref = (lam_ref - x).clamp(min=1e-6).pow(-sq)
    o_ref = (w_ref.to(v.dtype).float() @ v.float())

    # --- raw kernel launch (branch signature: H after N_CTX)
    o = torch.empty_like(q)
    lam = torch.empty((B * H, N), device=DEVICE, dtype=torch.float32)
    d_sum = torch.empty((B * H, N), device=DEVICE, dtype=torch.float32)
    argmax = torch.empty((B * H, N), device=DEVICE, dtype=torch.int32)
    wsum = torch.empty((B * H, N), device=DEVICE, dtype=torch.float32)
    lambda_init = torch.full((N,), 1.1, device=DEVICE, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    sfa._stieltjes_attn_fwd[grid](
        q, k, v, o, lam, d_sum, argmax, wsum, lambda_init,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale, N, H,
        sq=sq, NUM_ITER=5, HALLEY=False, EPS=1e-6,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=False, NORMALIZE=False,
    )

    lam_k = lam.view(N)
    lam_r = lam_ref.view(N)
    dlam = (lam_k - lam_r).abs()
    ws_k = wsum.view(N)
    ws_r = w_ref.sum(-1).view(N)
    do_ = (o.float() - o_ref).abs().amax(dim=-1).view(N)

    print(f"lambda:  max|d| {dlam.max():.6f}  mean {dlam.mean():.6f}")
    print(f"sum_w:   kernel [{ws_k.min():.4f},{ws_k.max():.4f}]  "
          f"ref [{ws_r.min():.4f},{ws_r.max():.4f}]  "
          f"max|d| {(ws_k-ws_r).abs().max():.6f}")
    print(f"output:  max row err {do_.max():.4f}")
    worst = int(do_.argmax())
    print(f"worst row {worst}: lam_k {lam_k[worst]:.6f} lam_r "
          f"{lam_r[worst]:.6f}  ws_k {ws_k[worst]:.4f} ws_r "
          f"{ws_r[worst]:.4f}")
    # NUM_ITER sensitivity: rerun ref at 5 iters (matching kernel)
    lam5 = torch.full_like(s_max, 1.1)
    for _ in range(5):
        diff = (lam5 - x).clamp(min=1e-6)
        f = diff.pow(-sq).sum(-1, keepdim=True) - 1.0
        fd = -sq * diff.pow(-sq - 1.0).sum(-1, keepdim=True)
        lam5 = lam5 - f / fd
    print(f"ref@5-vs-ref@10 lambda max|d|: "
          f"{(lam5 - lam_ref).abs().max():.6f}  <- test compares "
          f"kernel@5 to ref@10; if large, the TEST is at fault")


if __name__ == "__main__":
    main()
