"""Compare Triton stieltjes_attention forward output against the PyTorch ref
on random inputs across the shapes used in training. Reports max abs diff and
max rel diff. Intent: figure out why Triton fwd on trained needle weights gives
garbage accuracy while PyTorch ref gives 0.92.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref  # noqa: E402

DEVICE = torch.device("cuda")

# Match the nanoGPT model shapes used in training:
# B (batch), H=6, D=64 (n_embd=384, 6 heads), N varies.
CASES = [
    # (B, H, N, D, dtype, causal, q, num_iter)
    # Diagnostic: does bumping num_iter fix the causal-q=4 divergence?
    (1, 6, 2048, 64, torch.float32, True,  4.0,  3),
    (1, 6, 2048, 64, torch.float32, True,  4.0,  5),
    (1, 6, 2048, 64, torch.float32, True,  4.0, 10),
    (1, 6, 2048, 64, torch.float32, True,  4.0, 20),
    (4, 6, 2048, 64, torch.float32, True,  4.0,  3),
    (4, 6, 2048, 64, torch.float32, True,  4.0, 10),
    (1, 6, 2048, 64, torch.float32, False, 4.0,  3),
    (1, 6, 2048, 64, torch.float32, False, 4.0, 10),
]


def run_case(B, H, N, D, dtype, causal, q, num_iter=3):
    sm_scale = 1.0 / (D ** 0.5)
    torch.manual_seed(0)
    q_t = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k_t = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v_t = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        # Ref uses num_iter=5 by default; bump to match whatever Triton uses
        # so we compare apples-to-apples.
        out_ref = stieltjes_attention_ref(
            q_t, k_t, v_t, sm_scale=sm_scale, causal=causal, stieltjes_q=q,
            num_iter=max(num_iter, 5),
        )
        out_tri = stieltjes_attention(
            q_t, k_t, v_t, causal=causal, sm_scale=sm_scale, stieltjes_q=q,
            num_iter=num_iter,
        )

    # Upcast for comparison
    ref = out_ref.float()
    tri = out_tri.float()
    diff = (ref - tri).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref.abs() + 1e-6)).max().item()
    mean_abs = diff.mean().item()
    # Also report "sign agreement" — does argmax-over-last-dim agree?
    # (proxy for classifier-head behavior)
    ref_argmax = ref.argmax(dim=-1)
    tri_argmax = tri.argmax(dim=-1)
    argmax_agree = (ref_argmax == tri_argmax).float().mean().item()
    return dict(
        max_abs=max_abs, max_rel=max_rel, mean_abs=mean_abs,
        argmax_agree=argmax_agree, ref_mean=ref.abs().mean().item(),
    )


def main():
    print(f"{'B':>2s} {'H':>2s} {'N':>6s} {'D':>3s}  {'dtype':>8s} {'C':>1s} {'q':>3s} {'it':>3s}  "
          f"{'max_abs':>10s} {'mean_abs':>10s} {'argmax_agree':>12s} {'ref_mean':>10s}")
    for case in CASES:
        B, H, N, D, dtype, causal, q, num_iter = case
        r = run_case(B, H, N, D, dtype, causal, q, num_iter=num_iter)
        dname = str(dtype).replace("torch.", "")
        print(f"{B:>2d} {H:>2d} {N:>6d} {D:>3d}  {dname:>8s} {int(causal):>1d} {q:>3.1f} {num_iter:>3d}  "
              f"{r['max_abs']:>10.5f} {r['mean_abs']:>10.5f} "
              f"{r['argmax_agree']:>12.4f} {r['ref_mean']:>10.5f}")


if __name__ == "__main__":
    main()
