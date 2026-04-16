"""Numerically demonstrate where Triton stj diverges from PyTorch ref across N.

Tests seq lengths {128, 256, 512, 1024, 2048, 4096, 8192} with realistic shapes
(B=2, H=6, D=64) at q=4 causal — same shapes as our trained models. Reports
max-abs, mean-abs, and per-row max-abs (worst row) at each N.

If Triton == ref everywhere: the U-shape in eval is in the model code, not the
kernel. If Triton diverges sharply at certain N: that IS the U-shape mechanism.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref  # noqa

DEVICE = torch.device("cuda")
B, H, D = 2, 6, 64
SEQS = [128, 256, 512, 1024, 2048, 4096, 8192]


def run_case(N, q, num_iter, dtype=torch.float32, causal=True):
    sm_scale = 1.0 / (D ** 0.5)
    torch.manual_seed(0)
    q_t = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k_t = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v_t = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    with torch.no_grad():
        out_ref = stieltjes_attention_ref(
            q_t, k_t, v_t, sm_scale=sm_scale, causal=causal,
            stieltjes_q=q, num_iter=max(num_iter, 5),
        )
        out_tri = stieltjes_attention(
            q_t, k_t, v_t, causal=causal, sm_scale=sm_scale,
            stieltjes_q=q, num_iter=num_iter,
        )
    diff = (out_ref.float() - out_tri.float()).abs()
    # per-row worst (across batch, head, position)
    per_row_max = diff.amax(dim=-1)  # (B, H, N)
    return dict(
        max_abs=diff.max().item(),
        mean_abs=diff.mean().item(),
        worst_row_max=per_row_max.max().item(),
        argmax_agree=(out_ref.float().argmax(-1) == out_tri.float().argmax(-1)).float().mean().item(),
        ref_norm=out_ref.float().abs().mean().item(),
    )


def main():
    print(f"shape: B={B} H={H} D={D}, q=4.0, causal=True, fp32, num_iter=10")
    print(f"{'N':>6}  {'max_abs':>10}  {'mean_abs':>10}  {'worst_row':>10}  {'argmax_agree':>12}  {'ref_norm':>10}")
    for N in SEQS:
        try:
            r = run_case(N, q=4.0, num_iter=10)
            print(f"{N:>6}  {r['max_abs']:>10.5f}  {r['mean_abs']:>10.5f}  "
                  f"{r['worst_row_max']:>10.5f}  {r['argmax_agree']:>12.4f}  "
                  f"{r['ref_norm']:>10.5f}")
        except Exception as e:
            print(f"{N:>6}  FAIL {e}")


if __name__ == "__main__":
    main()
