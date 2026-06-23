"""
Thorough correctness suite for the Triton normalized Stieltjes mode
(normalize=True): outputs and gradients must match the PyTorch reference
(Jack's normalized `stieltjes`, architecture_new branch).

PART A — FORWARD vs Jack's mapping on raw (B,H,N,D) tensors.
  N ∈ {16, 32, 64, 96, 128, 256} (exercises padded query rows AND padded key
  columns), q ∈ {2,4,8,16}, fp32; fp16 at multiples of 64. Non-causal compares
  against scores → JackNorm → @v. Causal compares against the unnormalized ref
  with causal mask + explicit renormalization.
  PASS: fp32 max|Δ| ≤ 5e-4 (expected ~2e-5), no NaN.

PART B — BACKWARD vs PyTorch autograd through Jack's mapping (fp32).
  Same pipeline both sides: scores = qk^T·scale → mapping → @v → backward(do).
  N ∈ {16, 64, 128}, q ∈ {2,4,16}, D ∈ {64, 128}.
  PASS: dQ/dK/dV rel ≤ 2e-2 (expected ~1e-3), no NaN.

PART C — NaN sweep for normalize mode: N ∈ {32,64,128}, fp16/fp32, q ∈ {2,4,16}:
  no NaN in O/dQ/dK/dV (padding hygiene under the new mode).
"""
import sys
import torch

WORKTREE = "/users/PAS2402/alexg/softmax/psm-architecture-new"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")

from mappings.stieltjes import StieltjesTransform as JackNorm  # noqa: E402
from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref  # noqa: E402

FAILURES = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not cond:
        FAILURES.append(name)


def ref_norm_attention(q, k, v, sm_scale, sq, causal):
    """PyTorch reference: normalized stieltjes attention (Jack's mapping)."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        N = scores.shape[-1]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        # Jack's mapping has no causal concept; mask + renormalize explicitly.
        w_ref = stieltjes_attention_ref(q, k, v, sm_scale, causal=True,
                                        stieltjes_q=sq, num_iter=25, eps=1e-6)
        # ref returns UNNORMALIZED output; rebuild weights to renormalize:
        # recompute unnormalized weights the same way the ref does
        s_max = scores.max(dim=-1, keepdim=True).values
        x = scores - s_max
        lambd = torch.full_like(s_max, 1.1)
        for _ in range(25):
            diff = (lambd - x).clamp(min=1e-6)
            f_val = diff.pow(-sq).sum(dim=-1, keepdim=True) - 1.0
            f_deriv = -sq * diff.pow(-sq - 1.0).sum(dim=-1, keepdim=True)
            lambd = lambd - f_val / f_deriv
        w = (lambd - x).clamp(min=1e-6).pow(-sq)
        w = w.masked_fill(~mask, 0.0)
        p = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.matmul(p, v)
    mapping = JackNorm(q=sq, num_iter=30, eps=1e-9)
    p = mapping.translate_logits(scores, dim=-1)
    return torch.matmul(p, v)


def part_a(device):
    print("=" * 78)
    print("PART A — FORWARD: Triton(normalize=True) vs PyTorch normalized reference")
    print("=" * 78)
    B, H, D = 2, 4, 64
    for causal in [False, True]:
        for N in [16, 32, 64, 96, 128, 256]:
            for sq in [2.0, 4.0, 8.0, 16.0]:
                torch.manual_seed(0)
                sm = 1.0 / D ** 0.5
                qd = torch.randn(B, H, N, D, device=device)
                kd = torch.randn(B, H, N, D, device=device)
                vd = torch.randn(B, H, N, D, device=device)
                ref = ref_norm_attention(qd, kd, vd, sm, sq, causal)
                tri = stieltjes_attention(qd, kd, vd, causal=causal, sm_scale=sm,
                                          stieltjes_q=sq, num_iter=25,
                                          normalize=True)
                err = (tri - ref).abs().max().item()
                nan = torch.isnan(tri).any().item()
                check(f"fwd causal={causal} N={N:3d} q={sq:>4.0f}",
                      err <= 5e-4 and not nan,
                      f"max|Δ|={err:.3e} nan={nan}")
    # fp16 spot checks (non-causal, multiples of 64)
    for N in [64, 128]:
        for sq in [2.0, 4.0]:
            torch.manual_seed(0)
            sm = 1.0 / D ** 0.5
            qd = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
            kd = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
            vd = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
            ref = ref_norm_attention(qd.float(), kd.float(), vd.float(), sm, sq, False)
            tri = stieltjes_attention(qd, kd, vd, causal=False, sm_scale=sm,
                                      stieltjes_q=sq, num_iter=25,
                                      normalize=True).float()
            err = (tri - ref).abs().max().item()
            nan = torch.isnan(tri).any().item()
            check(f"fwd fp16 N={N:3d} q={sq:>4.0f}", err <= 5e-3 and not nan,
                  f"max|Δ|={err:.3e} nan={nan}")


def part_b(device):
    print("\n" + "=" * 78)
    print("PART B — BACKWARD: Triton(normalize=True) vs PyTorch autograd (fp32)")
    print("=" * 78)
    for D in [64, 128]:
        for N in [16, 64, 128]:
            for sq in [2.0, 4.0, 16.0]:
                torch.manual_seed(7)
                B, H = 2, 2
                sm = 1.0 / D ** 0.5
                qf = torch.randn(B, H, N, D, device=device)
                kf = torch.randn(B, H, N, D, device=device)
                vf = torch.randn(B, H, N, D, device=device)

                # PyTorch reference grads (autograd through Jack's mapping)
                q1 = qf.clone().requires_grad_(True)
                k1 = kf.clone().requires_grad_(True)
                v1 = vf.clone().requires_grad_(True)
                o1 = ref_norm_attention(q1, k1, v1, sm, sq, False)
                do = torch.randn_like(o1)
                o1.backward(do)

                # Triton normalize-mode grads
                q2 = qf.clone().requires_grad_(True)
                k2 = kf.clone().requires_grad_(True)
                v2 = vf.clone().requires_grad_(True)
                o2 = stieltjes_attention(q2, k2, v2, causal=False, sm_scale=sm,
                                         stieltjes_q=sq, num_iter=25,
                                         normalize=True)
                o2.backward(do)

                rels = []
                nan = False
                for g1, g2 in [(q1.grad, q2.grad), (k1.grad, k2.grad),
                               (v1.grad, v2.grad)]:
                    nan = nan or torch.isnan(g2).any().item()
                    rels.append((g2 - g1).abs().max().item()
                                / (g1.abs().max().item() + 1e-12))
                worst = max(rels)
                check(f"bwd D={D:3d} N={N:3d} q={sq:>4.0f}",
                      worst <= 2e-2 and not nan,
                      f"rel dQ={rels[0]:.2e} dK={rels[1]:.2e} dV={rels[2]:.2e} nan={nan}")


def part_c(device):
    print("\n" + "=" * 78)
    print("PART C — NaN sweep (normalize mode, padding shapes)")
    print("=" * 78)
    D = 64
    for N in [32, 64, 128]:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            for sq in [2.0, 4.0, 16.0]:
                torch.manual_seed(0)
                sm = 1.0 / D ** 0.5
                qd = torch.randn(1, 1, N, D, device=device, dtype=dtype,
                                 requires_grad=True)
                kd = torch.randn(1, 1, N, D, device=device, dtype=dtype,
                                 requires_grad=True)
                vd = torch.randn(1, 1, N, D, device=device, dtype=dtype,
                                 requires_grad=True)
                o = stieltjes_attention(qd, kd, vd, causal=False, sm_scale=sm,
                                        stieltjes_q=sq, num_iter=15,
                                        normalize=True)
                o.backward(torch.randn_like(o))
                nan = (torch.isnan(o).any().item()
                       or torch.isnan(qd.grad).any().item()
                       or torch.isnan(kd.grad).any().item()
                       or torch.isnan(vd.grad).any().item())
                check(f"nan-sweep N={N:3d} {dtype_name} q={sq:>4.0f}", not nan,
                      f"nan={nan}")


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    part_a(device)
    part_b(device)
    part_c(device)
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("RESULT: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
