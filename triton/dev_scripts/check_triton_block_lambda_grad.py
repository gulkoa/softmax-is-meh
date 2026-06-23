"""
Correctness test for the new Triton kernel block_lambda_grad option.

Compares:
  Triton(block_lambda_grad=False)  vs PyTorch NR autograd (IFT)
  Triton(block_lambda_grad=True)   vs PyTorch BS autograd

Reports max|Δ| in dQ, dK, dV for each.
"""
import sys

import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")

from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref  # noqa: E402
from maxretr_bs_vs_nr import StieltjesBSTransform, StieltjesNRTransform  # noqa: E402


def stieltjes_via_mapping(mapping, q, k, v, sm_scale, causal=False):
    """Compute attention output via a ProbabilitySimplexMapping."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        N = scores.shape[-1]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = mapping.translate_logits(scores, dim=-1)
    return torch.matmul(weights, v), weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 64

    for q in [1.0, 4.0, 16.0]:
        print(f"\n{'='*72}")
        print(f"q = {q}")
        print(f"{'='*72}")

        sm_scale = 1.0 / (D ** 0.5)

        # Reference fp32 inputs
        q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        # ---- BS reference (PyTorch bisection w/ autograd) ----
        bs_map = StieltjesBSTransform(q=q, num_iter=64, eps=1e-9)
        q_bs = q_ref.clone().requires_grad_(True)
        k_bs = k_ref.clone().requires_grad_(True)
        v_bs = v_ref.clone().requires_grad_(True)
        o_bs, _ = stieltjes_via_mapping(bs_map, q_bs, k_bs, v_bs, sm_scale)
        do = torch.randn_like(o_bs)
        o_bs.backward(do)
        dq_bs, dk_bs, dv_bs = q_bs.grad.clone(), k_bs.grad.clone(), v_bs.grad.clone()

        # ---- NR reference (PyTorch NR w/ autograd through iteration) ----
        nr_map = StieltjesNRTransform(q=q, num_iter=20, eps=1e-9)
        q_nr = q_ref.clone().requires_grad_(True)
        k_nr = k_ref.clone().requires_grad_(True)
        v_nr = v_ref.clone().requires_grad_(True)
        o_nr, _ = stieltjes_via_mapping(nr_map, q_nr, k_nr, v_nr, sm_scale)
        o_nr.backward(do)
        dq_nr, dk_nr, dv_nr = q_nr.grad.clone(), k_nr.grad.clone(), v_nr.grad.clone()

        # ---- Triton with block_lambda_grad=False (IFT) ----
        q_t1 = q_ref.clone().to(torch.float16).requires_grad_(True)
        k_t1 = k_ref.clone().to(torch.float16).requires_grad_(True)
        v_t1 = v_ref.clone().to(torch.float16).requires_grad_(True)
        o_t1 = stieltjes_attention(q_t1, k_t1, v_t1, causal=False,
                                   sm_scale=sm_scale, stieltjes_q=q,
                                   num_iter=10, block_lambda_grad=False)
        o_t1.backward(do.to(torch.float16))
        dq_t1, dk_t1, dv_t1 = (q_t1.grad.float(), k_t1.grad.float(), v_t1.grad.float())

        # ---- Triton with block_lambda_grad=True (BS-style) ----
        q_t2 = q_ref.clone().to(torch.float16).requires_grad_(True)
        k_t2 = k_ref.clone().to(torch.float16).requires_grad_(True)
        v_t2 = v_ref.clone().to(torch.float16).requires_grad_(True)
        o_t2 = stieltjes_attention(q_t2, k_t2, v_t2, causal=False,
                                   sm_scale=sm_scale, stieltjes_q=q,
                                   num_iter=10, block_lambda_grad=True)
        o_t2.backward(do.to(torch.float16))
        dq_t2, dk_t2, dv_t2 = (q_t2.grad.float(), k_t2.grad.float(), v_t2.grad.float())

        # ---- Forward comparison ----
        print(f"forward max|Triton - BS|     = {(o_t1.float() - o_bs).abs().max().item():.3e}")
        print(f"forward max|Triton - NR|     = {(o_t1.float() - o_nr).abs().max().item():.3e}")

        # ---- Backward comparison: Triton IFT vs PyTorch NR (should match) ----
        print(f"\nTriton(block_lambda_grad=False) vs PyTorch NR (IFT):")
        print(f"  max|dQ|: {(dq_t1 - dq_nr).abs().max().item():.3e}  (rel: {(dq_t1 - dq_nr).abs().max().item()/dq_nr.abs().max().item():.3e})")
        print(f"  max|dK|: {(dk_t1 - dk_nr).abs().max().item():.3e}  (rel: {(dk_t1 - dk_nr).abs().max().item()/dk_nr.abs().max().item():.3e})")
        print(f"  max|dV|: {(dv_t1 - dv_nr).abs().max().item():.3e}  (rel: {(dv_t1 - dv_nr).abs().max().item()/dv_nr.abs().max().item():.3e})")

        # ---- Backward comparison: Triton BS-mode vs PyTorch BS (should match) ----
        print(f"\nTriton(block_lambda_grad=True) vs PyTorch BS:")
        print(f"  max|dQ|: {(dq_t2 - dq_bs).abs().max().item():.3e}  (rel: {(dq_t2 - dq_bs).abs().max().item()/dq_bs.abs().max().item():.3e})")
        print(f"  max|dK|: {(dk_t2 - dk_bs).abs().max().item():.3e}  (rel: {(dk_t2 - dk_bs).abs().max().item()/dk_bs.abs().max().item():.3e})")
        print(f"  max|dV|: {(dv_t2 - dv_bs).abs().max().item():.3e}  (rel: {(dv_t2 - dv_bs).abs().max().item()/dv_bs.abs().max().item():.3e})")

        # ---- Cross-comparison: Triton BS vs Triton IFT (should DIFFER) ----
        print(f"\nTriton(BS) vs Triton(IFT) — should DIFFER (sanity):")
        print(f"  max|dQ|: {(dq_t2 - dq_t1).abs().max().item():.3e}")
        print(f"  max|dK|: {(dk_t2 - dk_t1).abs().max().item():.3e}")
        print(f"  max|dV|: {(dv_t2 - dv_t1).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
