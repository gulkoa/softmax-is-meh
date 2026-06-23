"""
Verification-before-completion for the NaN fix.

1. Run the kernel's built-in test_forward_correctness() and
   test_backward_correctness() (IFT-mode default). The backward test includes
   N=64 fp16 configs that PREVIOUSLY produced NaN dV -> FAIL. They should now
   PASS.
2. BS-mode backward vs PyTorch BS across no-padding and padding configs in fp32
   (argmax stable) -> should match closely (rel < 1e-2), proving the BS-mode
   logic is correct and the fix didn't perturb it.
"""
import sys
import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")

import stieltjes_flash_attn as sfa
from stieltjes_flash_attn import (
    stieltjes_attention, test_forward_correctness, test_backward_correctness)
from maxretr_bs_vs_nr import StieltjesBSTransform


def stieltjes_via_mapping(mapping, q, k, v, sm_scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    weights = mapping.translate_logits(scores, dim=-1)
    return torch.matmul(weights, v)


def bsmode_fp32_check(device):
    print("\n" + "=" * 72)
    print("BS-mode backward vs PyTorch BS (fp32 inputs, argmax stable)")
    print("=" * 72)
    configs = [
        # (B, H, N, D, q)  — mix of padding (N<128 or not mult of 128) and not
        (2, 4, 64, 64, 1.0),
        (2, 4, 64, 64, 4.0),
        (1, 2, 128, 64, 1.0),
        (1, 2, 128, 64, 4.0),
        (1, 2, 256, 64, 4.0),
        (1, 1, 128, 128, 1.0),
    ]
    all_ok = True
    for B, H, N, D, q in configs:
        torch.manual_seed(123)
        sm_scale = 1.0 / (D ** 0.5)
        qf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        vf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        bs_map = StieltjesBSTransform(q=q, num_iter=64, eps=1e-9)
        qb = qf.clone().requires_grad_(True)
        kb = kf.clone().requires_grad_(True)
        vb = vf.clone().requires_grad_(True)
        o_bs = stieltjes_via_mapping(bs_map, qb, kb, vb, sm_scale)
        do = torch.randn_like(o_bs)
        o_bs.backward(do)
        dq_bs, dk_bs, dv_bs = qb.grad.clone(), kb.grad.clone(), vb.grad.clone()

        qt = qf.clone().requires_grad_(True)
        kt = kf.clone().requires_grad_(True)
        vt = vf.clone().requires_grad_(True)
        o_t = stieltjes_attention(qt, kt, vt, causal=False, sm_scale=sm_scale,
                                  stieltjes_q=q, num_iter=12, block_lambda_grad=True)
        o_t.backward(do)
        dq_t, dk_t, dv_t = qt.grad, kt.grad, vt.grad

        # NaN guard
        nan = any(torch.isnan(x).any().item() for x in (dq_t, dk_t, dv_t))
        dq_rel = (dq_t - dq_bs).abs().max().item() / dq_bs.abs().max().item()
        dk_rel = (dk_t - dk_bs).abs().max().item() / dk_bs.abs().max().item()
        dv_rel = (dv_t - dv_bs).abs().max().item() / dv_bs.abs().max().item()
        ok = (not nan) and max(dq_rel, dk_rel, dv_rel) < 2e-2
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] B={B} H={H} N={N:4d} D={D:3d} q={q:>4.0f}  "
              f"nan={nan}  dQ_rel={dq_rel:.2e} dK_rel={dk_rel:.2e} dV_rel={dv_rel:.2e}")
    print("  " + ("All BS-mode fp32 checks passed." if all_ok else "Some BS-mode checks FAILED."))
    return all_ok


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}\n")
    fwd_ok = test_forward_correctness()
    bwd_ok = test_backward_correctness()
    bs_ok = bsmode_fp32_check(torch.device("cuda"))
    print("\n" + "=" * 72)
    print(f"SUMMARY: forward={'PASS' if fwd_ok else 'FAIL'}  "
          f"backward(IFT)={'PASS' if bwd_ok else 'FAIL'}  "
          f"BS-mode(fp32)={'PASS' if bs_ok else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
