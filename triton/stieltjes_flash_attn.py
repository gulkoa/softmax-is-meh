"""
Stieltjes Flash Attention
=========================

Memory-efficient attention using the Stieltjes transform instead of softmax.

Standard attention:  O = softmax(QK^T / sqrt(d)) @ V
Stieltjes attention: O = stieltjes(QK^T / sqrt(d)) @ V

where stieltjes(s)_j = (λ - s_j)^{-q} with λ chosen so Σ_j (λ - s_j)^{-q} = 1.

The Triton kernel uses a multi-pass tiled approach (flash-style) to avoid
materializing the N×N attention matrix:

  Pass 1: Row-wise max of QK^T scores          (1 sweep over K)
  Pass 2: Newton-Raphson for λ                  (num_iter sweeps over K)
  Pass 3: Compute output weights, accumulate PV (1 sweep over K & V)

Total: (2 + num_iter) matmul sweeps.  Memory: O(N·d) not O(N²).
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# PyTorch reference implementation
# ---------------------------------------------------------------------------

def stieltjes_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
    stieltjes_q: float = 1.0,
    num_iter: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reference (non-flash) Stieltjes attention in PyTorch.

    Args:
        q, k, v: (B, H, N, D)  query / key / value
        sm_scale: scaling factor (typically 1/sqrt(d))
        causal: apply causal mask
        stieltjes_q: order of the Stieltjes transform
        num_iter: Newton-Raphson iterations
        eps: numerical stability

    Returns:
        o: (B, H, N, D)
    """
    # scores: (B, H, N, N)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if causal:
        N = scores.shape[-1]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, -1e6)

    # --- Stieltjes normalization along last dim ---
    sq = stieltjes_q
    s_max = scores.max(dim=-1, keepdim=True).values
    x = scores - s_max  # centered; max = 0

    n_cols = scores.shape[-1]
    lambd = torch.full_like(s_max, float(n_cols) ** (1.0 / sq))

    for _ in range(num_iter):
        diff = (lambd - x).clamp(min=eps)
        f_val = diff.pow(-sq).sum(dim=-1, keepdim=True) - 1.0
        f_deriv = -sq * diff.pow(-sq - 1.0).sum(dim=-1, keepdim=True)
        lambd = torch.maximum(lambd - f_val / f_deriv, lambd * 0.5)

    diff = (lambd - x).clamp(min=eps)
    weights = diff.pow(-sq)  # (B, H, N, N), rows sum to ~1

    o = torch.matmul(weights.to(v.dtype), v)
    return o


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _stieltjes_attn_fwd(
    Q, K, V, O,
    Lambda,  # (B*H, N) — stores λ per query row for backward
    D_sum,   # (B*H, N) — stores Σ(λ-s)^{-q-1} per query row for backward
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale,
    N_CTX,
    sq: tl.constexpr,        # Stieltjes q parameter
    NUM_ITER: tl.constexpr,  # Newton-Raphson iterations
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # off_hz indexes into the flattened (B, H) dims.
    # For contiguous (B, H, N, D) tensors: stride_qz = H*N*D, stride_qh = N*D.
    # We need: byte offset = off_z * stride_qz + off_h * stride_qh.
    # But since stride_qz = H * stride_qh for contiguous tensors and
    # off_hz = off_z * H + off_h, we can compute the combined offset as:
    # off_z * H * stride_qh + off_h * stride_qh = off_hz * stride_qh.
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    # -- load Q block --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # ===== PASS 1: Row-wise max of QK^T =====
    row_max = tl.full([BLOCK_M], value=-1e30, dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q_block, tl.trans(k_block)) * sm_scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, -1e30)

        tile_max = tl.max(qk, axis=1)
        row_max = tl.maximum(row_max, tile_max)

    # ===== PASS 2: Newton-Raphson for λ =====
    # After centering by row_max, scores are ≤ 0 and λ must be > 0.
    # Init: n^{1/q} is exact for uniform scores.
    lambd = tl.full([BLOCK_M], value=float(N_CTX), dtype=tl.float32)
    # For causal, effective n_cols varies per row but N_CTX^{1/q} is a safe overestimate
    # that NR will quickly correct.

    for _nr in tl.static_range(NUM_ITER):
        f_val = tl.zeros([BLOCK_M], dtype=tl.float32)
        f_deriv = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

            qk = tl.dot(q_block, tl.trans(k_block)) * sm_scale

            if CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, -1e30)

            centered = qk - row_max[:, None]
            diff = tl.maximum(lambd[:, None] - centered, EPS)

            if sq == 1.0:
                inv_q = 1.0 / diff
                inv_q1 = inv_q * inv_q
            else:
                log_diff = tl.log(diff)
                inv_q = tl.exp(log_diff * (-sq))
                inv_q1 = tl.exp(log_diff * (-sq - 1.0))

            # Masked positions have centered ≈ -1e30, diff ≈ lambd+1e30 → inv ≈ 0
            f_val += tl.sum(inv_q, axis=1)
            f_deriv += tl.sum(inv_q1, axis=1)

        # f(λ) = Σ(λ-x)^{-q} - 1,  f'(λ) = -q Σ(λ-x)^{-q-1}
        f_val = f_val - 1.0
        f_deriv = f_deriv * (-sq)
        lambd = tl.maximum(lambd - f_val / f_deriv, lambd * 0.5)

    # ===== PASS 3: Compute attention output P @ V =====
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    d_sum = tl.zeros([BLOCK_M], dtype=tl.float32)  # Σ(λ-s)^{-q-1} for backward

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v_block = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q_block, tl.trans(k_block)) * sm_scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, -1e30)

        centered = qk - row_max[:, None]
        diff = tl.maximum(lambd[:, None] - centered, EPS)

        if sq == 1.0:
            weights = 1.0 / diff
            d_tile = weights * weights  # (λ-s)^{-2} = (λ-s)^{-q-1} for q=1
        else:
            log_diff = tl.log(diff)
            weights = tl.exp(log_diff * (-sq))
            d_tile = tl.exp(log_diff * (-sq - 1.0))

        d_sum += tl.sum(d_tile, axis=1)

        # Accumulate P @ V
        acc += tl.dot(weights.to(v_block.dtype), v_block)

    # Store output
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(q_block.dtype), mask=offs_m[:, None] < N_CTX)

    # Store λ and D for backward
    lambda_ptrs = Lambda + off_hz * N_CTX + offs_m
    d_ptrs = D_sum + off_hz * N_CTX + offs_m
    tl.store(lambda_ptrs, lambd + row_max, mask=offs_m < N_CTX)  # store absolute λ
    tl.store(d_ptrs, d_sum, mask=offs_m < N_CTX)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------

class StieltjesAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, stieltjes_q=1.0, num_iter=5):
        B, H, N, D = q.shape
        assert k.shape == v.shape == (B, H, N, D)
        assert D in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        lam = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
        d_sum = torch.empty((B * H, N), device=q.device, dtype=torch.float32)

        # Flatten batch and head dims for stride computation
        # q is (B, H, N, D) contiguous
        BLOCK_M = 64
        BLOCK_N = 64

        grid = (triton.cdiv(N, BLOCK_M), B * H)

        _stieltjes_attn_fwd[grid](
            q, k, v, o,
            lam, d_sum,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            sm_scale,
            N,
            sq=stieltjes_q,
            NUM_ITER=num_iter,
            EPS=1e-6,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, o, lam, d_sum)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.stieltjes_q = stieltjes_q
        ctx.num_iter = num_iter
        return o

    @staticmethod
    def backward(ctx, do):
        """
        Analytical backward for Stieltjes attention.

        Given P_ij = (λ_i - s_ij)^{-q}, the Jacobian is:
            dS_ij = q · r_ij · (dP_ij − δ_i)
        where r_ij = (λ_i - s_ij)^{-q-1},  δ_i = (Σ_k dP_ik · r_ik) / D_i,
        and D_i = Σ_k r_ik.

        We recompute scores to avoid O(N²) storage (flash-style).
        """
        q, k, v, o, lam, d_sum_saved = ctx.saved_tensors
        sq = ctx.stieltjes_q
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        eps = 1e-6

        B, H, N, D = q.shape

        # Recompute scores and weights (flash-style: tile by tile would be
        # ideal, but for correctness we do it densely here; a full Triton
        # backward kernel is left as future work)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale

        if causal:
            mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, -1e6)

        # Recover λ (stored as absolute = λ_relative + row_max)
        lam_abs = lam.view(B, H, N, 1).float()

        diff = (lam_abs - scores).clamp(min=eps)
        if sq == 1.0:
            weights = 1.0 / diff
            r = weights * weights  # (λ-s)^{-2}
        else:
            log_diff = diff.log()
            weights = (log_diff * (-sq)).exp()
            r = (log_diff * (-sq - 1.0)).exp()

        # dP = dO @ V^T
        dP = torch.matmul(do.float(), v.float().transpose(-2, -1))

        # D_i = Σ_j r_ij
        D_i = r.sum(dim=-1, keepdim=True)

        # δ_i = (Σ_j dP_ij · r_ij) / D_i
        delta = (dP * r).sum(dim=-1, keepdim=True) / D_i.clamp(min=eps)

        # dS = q · r · (dP - δ)
        dS = sq * r * (dP - delta)

        if causal:
            dS = dS.masked_fill(~mask, 0.0)

        # dQ = dS @ K · sm_scale
        dq = torch.matmul(dS.to(q.dtype), k) * sm_scale
        # dK = dS^T @ Q · sm_scale
        dk = torch.matmul(dS.transpose(-2, -1).to(k.dtype), q) * sm_scale
        # dV = P^T @ dO
        dv = torch.matmul(weights.transpose(-2, -1).to(v.dtype), do)

        return dq, dk, dv, None, None, None, None


def stieltjes_attention(q, k, v, causal=False, sm_scale=None, stieltjes_q=1.0, num_iter=5):
    """
    Stieltjes flash attention.

    Args:
        q, k, v: (B, H, N, D) — query, key, value tensors
        causal: whether to apply causal masking
        sm_scale: attention scale factor (default: 1/sqrt(D))
        stieltjes_q: order of the Stieltjes transform (default 1.0)
        num_iter: Newton-Raphson iterations (default 5)
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return StieltjesAttention.apply(q, k, v, causal, sm_scale, stieltjes_q, num_iter)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_forward_correctness():
    torch.manual_seed(42)
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    configs = [
        # (B, H, N, D, causal, q)
        (1, 1, 64,  64, False, 1.0),
        (1, 1, 64,  64, True,  1.0),
        (2, 4, 128, 64, False, 1.0),
        (2, 4, 128, 64, True,  1.0),
        (1, 2, 256, 64, False, 2.0),
        (1, 2, 256, 64, True,  2.0),
        (1, 1, 128, 128, False, 1.0),
        (1, 1, 128, 128, True,  1.0),
    ]

    print("Forward correctness tests")
    print("-" * 70)
    all_passed = True

    for B, H, N, D, causal, sq in configs:
        q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        sm_scale = 1.0 / (D ** 0.5)

        ref = stieltjes_attention_ref(
            q.float(), k.float(), v.float(),
            sm_scale, causal=causal, stieltjes_q=sq, num_iter=10, eps=1e-6,
        ).half()

        tri = stieltjes_attention(
            q, k, v, causal=causal, sm_scale=sm_scale,
            stieltjes_q=sq, num_iter=5,
        ).half()

        max_err = (tri - ref).abs().max().item()
        mean_err = (tri - ref).abs().mean().item()
        passed = max_err < 0.05  # relaxed for fp16 + iterative solver
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] B={B} H={H} N={N:4d} D={D:3d} causal={causal!s:5s} q={sq}  "
              f"max_err={max_err:.4f}  mean_err={mean_err:.6f}")

    print("-" * 70)
    if all_passed:
        print("All forward tests passed.")
    else:
        print("Some tests FAILED.")
    return all_passed


def test_backward_correctness():
    torch.manual_seed(42)
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    configs = [
        (1, 1, 64,  64, False, 1.0),
        (1, 1, 64,  64, True,  1.0),
        (2, 2, 128, 64, False, 1.0),
    ]

    print("\nBackward correctness tests (vs PyTorch autograd on reference)")
    print("-" * 70)
    all_passed = True

    for B, H, N, D, causal, sq in configs:
        sm_scale = 1.0 / (D ** 0.5)

        # Reference: use PyTorch autograd on dense reference implementation
        q_ref = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        v_ref = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        do = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)

        o_ref = stieltjes_attention_ref(q_ref, k_ref, v_ref, sm_scale, causal=causal,
                                        stieltjes_q=sq, num_iter=10, eps=1e-6)
        o_ref.backward(do)
        dq_ref = q_ref.grad.clone()
        dk_ref = k_ref.grad.clone()
        dv_ref = v_ref.grad.clone()

        # Triton forward + analytical backward
        q_tri = q_ref.detach().clone().to(torch.float16).requires_grad_(True)
        k_tri = k_ref.detach().clone().to(torch.float16).requires_grad_(True)
        v_tri = v_ref.detach().clone().to(torch.float16).requires_grad_(True)

        o_tri = stieltjes_attention(q_tri, k_tri, v_tri, causal=causal, sm_scale=sm_scale,
                                    stieltjes_q=sq, num_iter=5)
        o_tri.backward(do.to(torch.float16))
        dq_tri = q_tri.grad.float()
        dk_tri = k_tri.grad.float()
        dv_tri = v_tri.grad.float()

        dq_err = (dq_tri - dq_ref).abs().max().item()
        dk_err = (dk_tri - dk_ref).abs().max().item()
        dv_err = (dv_tri - dv_ref).abs().max().item()

        passed = max(dq_err, dk_err, dv_err) < 0.1  # fp16 tolerance
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] B={B} H={H} N={N:4d} D={D:3d} causal={causal!s:5s} q={sq}  "
              f"dQ_err={dq_err:.4f}  dK_err={dk_err:.4f}  dV_err={dv_err:.4f}")

    print("-" * 70)
    if all_passed:
        print("All backward tests passed.")
    else:
        print("Some backward tests FAILED.")
    return all_passed


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark():
    import triton.testing

    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(7, 13)],
            line_arg="provider",
            line_vals=["stieltjes-triton", "stieltjes-torch", "softmax-torch"],
            line_names=["Stieltjes (Triton)", "Stieltjes (PyTorch)", "Softmax (PyTorch)"],
            styles=[("red", "-"), ("blue", "--"), ("green", ":")],
            ylabel="TFLOPS",
            plot_name=f"stieltjes-flash-attention-fwd-B{B}-H{H}-D{D}",
            args={"B": B, "H": H, "D": D},
        )
        for B, H, D in [(4, 8, 64), (4, 8, 128)]
    ])
    def bench_fn(B, H, N_CTX, D, provider, device=DEVICE):
        dtype = torch.float16
        q = torch.randn(B, H, N_CTX, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N_CTX, D, dtype=dtype, device=device)
        v = torch.randn(B, H, N_CTX, D, dtype=dtype, device=device)
        sm_scale = 1.0 / (D ** 0.5)

        if provider == "stieltjes-triton":
            fn = lambda: stieltjes_attention(q, k, v, sm_scale=sm_scale)
        elif provider == "stieltjes-torch":
            fn = lambda: stieltjes_attention_ref(q.float(), k.float(), v.float(), sm_scale)
        elif provider == "softmax-torch":
            def fn():
                s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
                p = torch.softmax(s, dim=-1)
                return torch.matmul(p.half(), v)

        ms = triton.testing.do_bench(fn)
        # 2 matmuls: QK^T and PV, each 2*B*H*N*N*D flops
        # Stieltjes also recomputes QK^T for NR passes but we report standard flops
        flops = 2 * 2.0 * B * H * N_CTX * N_CTX * D
        return flops * 1e-12 / (ms * 1e-3)

    bench_fn.run(save_path=".", print_data=True)


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    fwd_ok = test_forward_correctness()
    bwd_ok = test_backward_correctness()
    if fwd_ok and bwd_ok:
        print("\n\nRunning benchmarks...\n")
        benchmark()
