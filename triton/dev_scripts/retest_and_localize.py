"""
(1) Retest the NaN fix, (2) localize the original 25% q=4 BS-mode error.

PART 1 — NaN fix retest: re-run the truth table. After masking padded query
rows in _stieltjes_score_helpers, EVERY cell should be NaN-free.

PART 2 — Localize the 25%: reproduce the EXACT inputs of
check_triton_block_lambda_grad.py (seed 42, q=1 then q=4 — so q=4 uses the
2nd RNG draw). Then:
  - count argmax mismatches between the Triton-stored argmax and torch.argmax
    of the fp32 scores, for THESE inputs;
  - find the (b,h,n) location of the max dQ error;
  - check whether that row is an argmax-mismatch row and report its top-2 score
    gap (near-tie indicator).
Hypothesis: the 25% is one (or few) near-tie rows where the fp16 forward argmax
differs from PyTorch's fp32 argmax, so the large kappa correction lands one
column off. This is inherent fp16 sensitivity of the max-centering gradient,
NOT a kernel logic bug.

PART 3 — Broad BS-mode correctness on no-padding configs (N multiple of 128),
to show the kernel matches PyTorch BS when there's neither padding nor near-tie
pathology.
"""
import sys
import torch
import triton

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")

import stieltjes_flash_attn as sfa
from stieltjes_flash_attn import stieltjes_attention
from maxretr_bs_vs_nr import StieltjesBSTransform


# ---------------------------------------------------------------------------
# PART 1 — NaN fix retest
# ---------------------------------------------------------------------------

def part1_nan_retest(device):
    print("=" * 84)
    print("PART 1 — NaN fix retest (every cell should now be NaN-free)")
    print("=" * 84)

    def check(N, dtype, q, blg):
        torch.manual_seed(0)
        B, H, D = 1, 1, 64
        sm_scale = 1.0 / (D ** 0.5)
        qd = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        kd = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        vd = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        o = stieltjes_attention(qd, kd, vd, causal=False, sm_scale=sm_scale,
                                stieltjes_q=q, num_iter=10, block_lambda_grad=blg)
        do = torch.randn_like(o)
        o.backward(do)
        return (torch.isnan(qd.grad).any().item()
                or torch.isnan(kd.grad).any().item()
                or torch.isnan(vd.grad).any().item()
                or torch.isnan(o).any().item())

    any_nan = False
    for blg in [False, True]:
        mode = "BS-mode" if blg else "IFT-mode"
        for N in [32, 64, 128, 256]:
            for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
                for q in [1.0, 4.0, 16.0]:
                    nan = check(N, dtype, q, blg)
                    any_nan = any_nan or nan
                    flag = "  <-- NaN!" if nan else ""
                    print(f"  {mode:8s} N={N:4d} {dtype_name} q={q:>4.0f}: nan={nan}{flag}")
    print(f"\n  >>> PART 1 RESULT: {'STILL HAS NaN' if any_nan else 'ALL CLEAN — NaN FIXED'}")
    return not any_nan


# ---------------------------------------------------------------------------
# Helper: forward kernel that also returns stored argmax
# ---------------------------------------------------------------------------

def fwd_with_argmax(q, k, v, sm_scale, sq, num_iter):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    lam = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    d_sum = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    argmax = torch.empty((B * H, N), device=q.device, dtype=torch.int32)
    lambda_init = torch.full((N,), 1.1, device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = (128, 64) if D <= 64 else (64, 64)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    sfa._stieltjes_attn_fwd[grid](
        q, k, v, o, lam, d_sum, argmax, lambda_init,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale, N, sq=sq, NUM_ITER=num_iter, EPS=1e-6,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=False,
    )
    return argmax.view(B, H, N)


def stieltjes_via_mapping(mapping, q, k, v, sm_scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    weights = mapping.translate_logits(scores, dim=-1)
    return torch.matmul(weights, v), weights


# ---------------------------------------------------------------------------
# PART 2 — Localize the original 25%
# ---------------------------------------------------------------------------

def part2_localize(device):
    print("\n" + "=" * 84)
    print("PART 2 — Localize the original 25% q=4 BS-mode error")
    print("  (reproducing check_triton_block_lambda_grad.py EXACT inputs: seed 42,")
    print("   q=1 then q=4, so q=4 uses the 2nd RNG draw)")
    print("=" * 84)

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 64
    sm_scale = 1.0 / (D ** 0.5)

    captured = {}
    for q in [1.0, 4.0]:
        q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        bs_map = StieltjesBSTransform(q=q, num_iter=64, eps=1e-9)
        q_bs = q_ref.clone().requires_grad_(True)
        k_bs = k_ref.clone().requires_grad_(True)
        v_bs = v_ref.clone().requires_grad_(True)
        o_bs, _ = stieltjes_via_mapping(bs_map, q_bs, k_bs, v_bs, sm_scale)
        do = torch.randn_like(o_bs)   # consume RNG exactly like the original
        if q == 4.0:
            captured = dict(q_ref=q_ref, k_ref=k_ref, v_ref=v_ref, do=do,
                            o_bs=o_bs, q_bs=q_bs, k_bs=k_bs)
        o_bs.backward(do)
        if q == 4.0:
            captured["dq_bs"] = q_bs.grad.clone()
            captured["dk_bs"] = k_bs.grad.clone()

    q_ref, k_ref, v_ref, do = (captured["q_ref"], captured["k_ref"],
                               captured["v_ref"], captured["do"])
    dq_bs = captured["dq_bs"]

    # Triton BS-mode (fp16), same as original
    q_t = q_ref.clone().to(torch.float16).requires_grad_(True)
    k_t = k_ref.clone().to(torch.float16).requires_grad_(True)
    v_t = v_ref.clone().to(torch.float16).requires_grad_(True)
    o_t = stieltjes_attention(q_t, k_t, v_t, causal=False, sm_scale=sm_scale,
                              stieltjes_q=4.0, num_iter=10, block_lambda_grad=True)
    o_t.backward(do.to(torch.float16))
    dq_t = q_t.grad.float()

    dq_diff = (dq_t - dq_bs).abs()
    max_err = dq_diff.max().item()
    rel = max_err / dq_bs.abs().max().item()
    print(f"\n  Reproduced dQ error: max={max_err:.3e}  rel={rel:.3e}")
    print(f"  (original check reported rel ~0.25 here)")

    # Argmax comparison for THESE inputs
    argmax_tri = fwd_with_argmax(q_t.detach(), k_t.detach(), v_t.detach(),
                                 sm_scale, 4.0, 10)
    scores_fp32 = (q_ref @ k_ref.transpose(-2, -1)) * sm_scale
    argmax_ref = scores_fp32.argmax(dim=-1).to(torch.int32)
    mism = (argmax_tri != argmax_ref)
    n_mism = mism.sum().item()
    print(f"\n  argmax mismatches (Triton fp16 vs torch fp32): {n_mism}/{argmax_ref.numel()}")

    # Locate the max dQ error
    loc = torch.nonzero(dq_diff == dq_diff.max())[0].tolist()
    b, h, n, d = loc
    print(f"  max dQ error at (b={b}, h={h}, query_row={n}, dim={d})")
    print(f"  is that row an argmax-mismatch row? {bool(mism[b, h, n].item())}")
    row_scores = scores_fp32[b, h, n]
    top2 = torch.topk(row_scores, 2).values
    gap = (top2[0] - top2[1]).item()
    print(f"  that row's top-2 score gap = {gap:.3e}  "
          f"(triton_argmax={argmax_tri[b,h,n].item()}, ref_argmax={argmax_ref[b,h,n].item()})")

    # For all mismatch rows, report top-2 gaps (are they all near-ties?)
    if n_mism > 0:
        print(f"\n  All {n_mism} mismatch rows — top-2 score gaps (near-tie if tiny):")
        midx = torch.nonzero(mism)
        for (bb, hh, nn) in midx.tolist():
            row = scores_fp32[bb, hh, nn]
            t2 = torch.topk(row, 2).values
            g = (t2[0] - t2[1]).item()
            ti = argmax_tri[bb, hh, nn].item()
            ri = argmax_ref[bb, hh, nn].item()
            # dQ error contribution at this row
            row_err = dq_diff[bb, hh, nn].max().item()
            print(f"    (b{bb},h{hh},row{nn}): gap={g:.3e} tri_am={ti} ref_am={ri} "
                  f"rowmax_dQ_err={row_err:.3e}")

    # Counterfactual: zero out the mismatch rows and recompute max error
    if n_mism > 0:
        mask_full = mism[:, :, :, None].expand_as(dq_diff)
        dq_diff_clean = dq_diff.clone()
        dq_diff_clean[mask_full] = 0.0
        print(f"\n  max dQ error EXCLUDING argmax-mismatch rows: "
              f"{dq_diff_clean.max().item():.3e}  "
              f"(rel {dq_diff_clean.max().item()/dq_bs.abs().max().item():.3e})")
        print("  -> if this is small, the 25% is ENTIRELY the near-tie argmax flips.")


# ---------------------------------------------------------------------------
# PART 3 — Broad BS-mode correctness on no-padding configs
# ---------------------------------------------------------------------------

def part3_broad(device):
    print("\n" + "=" * 84)
    print("PART 3 — BS-mode vs PyTorch BS on no-padding configs (N multiple of 128)")
    print("=" * 84)
    for N in [128, 256]:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            for q in [1.0, 4.0]:
                torch.manual_seed(7)
                B, H, D = 2, 4, 64
                sm_scale = 1.0 / (D ** 0.5)
                qf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
                kf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
                vf = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

                bs_map = StieltjesBSTransform(q=q, num_iter=64, eps=1e-9)
                qb = qf.clone().requires_grad_(True)
                kb = kf.clone().requires_grad_(True)
                vb = vf.clone().requires_grad_(True)
                o_bs, _ = stieltjes_via_mapping(bs_map, qb, kb, vb, sm_scale)
                do = torch.randn_like(o_bs)
                o_bs.backward(do)
                dq_bs, dk_bs = qb.grad.clone(), kb.grad.clone()

                qt = qf.clone().to(dtype).requires_grad_(True)
                kt = kf.clone().to(dtype).requires_grad_(True)
                vt = vf.clone().to(dtype).requires_grad_(True)
                o_t = stieltjes_attention(qt, kt, vt, causal=False, sm_scale=sm_scale,
                                          stieltjes_q=q, num_iter=10,
                                          block_lambda_grad=True)
                o_t.backward(do.to(dtype))
                dq_t, dk_t = qt.grad.float(), kt.grad.float()

                dq_rel = (dq_t - dq_bs).abs().max().item() / dq_bs.abs().max().item()
                dk_rel = (dk_t - dk_bs).abs().max().item() / dk_bs.abs().max().item()
                print(f"  N={N:4d} {dtype_name} q={q:>4.0f}: dQ_rel={dq_rel:.3e}  dK_rel={dk_rel:.3e}")


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    part1_nan_retest(device)
    part2_localize(device)
    part3_broad(device)


if __name__ == "__main__":
    main()
