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

DEVICE = torch.device("cuda")


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
        scores = scores.masked_fill(~mask, -1e30)

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

    if causal:
        weights = weights.masked_fill(~mask, 0.0)

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
    LAMBDA_INIT: tl.constexpr,  # precomputed N_CTX^{1/q} on host
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # off_hz indexes into the flattened (B, H) dims: off_hz = off_z * H + off_h.
    # Recover batch and head indices and compute base offsets using both z and h strides.
    # For standard contiguous (B, H, N, D) layout, stride_qz = H * stride_qh and
    # H_eff below equals the true number of heads, so this reduces to the original
    # formula q_offset = off_hz * stride_qh, etc.
    H_eff = stride_qz // stride_qh
    off_z = off_hz // H_eff
    off_h = off_hz % H_eff

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    # -- load Q block --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # ===== PASS 1: Row-wise max of QK^T =====
    row_max = tl.full([BLOCK_M], value=-1e30, dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
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
    lambd = tl.full([BLOCK_M], value=LAMBDA_INIT, dtype=tl.float32)
    # For causal, effective n_cols varies per row but N_CTX^{1/q} remains a safe
    # overestimate that NR will quickly correct.

    for _nr in tl.static_range(NUM_ITER):
        f_val = tl.zeros([BLOCK_M], dtype=tl.float32)
        f_deriv = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in tl.range(0, N_CTX, BLOCK_N):
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

    for start_n in tl.range(0, N_CTX, BLOCK_N):
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
# Triton backward kernels
# ---------------------------------------------------------------------------
#
# Backward for Stieltjes attention.  Given P_ij = (λ_i - s_ij)^{-q}:
#
#   r_ij  = (λ_i - s_ij)^{-q-1}          (derivative weight)
#   D_i   = Σ_j r_ij                      (saved from forward)
#   dP_ij = (dO @ V^T)_ij
#   δ_i   = (Σ_j dP_ij · r_ij) / D_i     (correction term)
#   dS_ij = q · r_ij · (dP_ij - δ_i)     (score gradient)
#
# Then: dQ = dS @ K · scale,  dK = dS^T @ Q · scale,  dV = P^T @ dO
#
# Three kernels, all recomputing scores on-the-fly (flash-style, O(N) memory):
#   1. _stieltjes_bwd_delta  — compute δ_i per query row
#   2. _stieltjes_bwd_dkdv   — compute dK, dV (iterate Q blocks for fixed K block)
#   3. _stieltjes_bwd_dq     — compute dQ     (iterate K blocks for fixed Q block)

@triton.jit
def _stieltjes_score_helpers(
    q_block, k_block, lam_row, sm_scale, offs_m, offs_n,
    sq: tl.constexpr, EPS: tl.constexpr, CAUSAL: tl.constexpr,
):
    """Recompute scores and Stieltjes weight helpers (P, r) for a Q×K tile.

    Returns (weights, r, qk) where:
      weights = (λ - s)^{-q}      — attention weights P
      r       = (λ - s)^{-q-1}    — derivative weights
      qk      = Q @ K^T * scale   — raw scores (for debugging / optional use)
    """
    qk = tl.dot(q_block, tl.trans(k_block)) * sm_scale

    if CAUSAL:
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, -1e30)

    diff = tl.maximum(lam_row[:, None] - qk, EPS)

    if sq == 1.0:
        weights = 1.0 / diff
        r = weights * weights
    else:
        log_diff = tl.log(diff)
        weights = tl.exp(log_diff * (-sq))
        r = tl.exp(log_diff * (-sq - 1.0))

    return weights, r, qk


@triton.jit
def _stieltjes_bwd_delta(
    Q, K, V, DO,
    Lambda, D_sum, Delta,          # Delta is the output: (B*H, N)
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    sm_scale, N_CTX,
    sq: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Compute δ_i = (Σ_j dP_ij · r_ij) / D_i for each query row."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    H_eff = stride_qz // stride_qh
    off_z = off_hz // H_eff
    off_h = off_hz % H_eff

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    do_off = off_z * stride_doz + off_h * stride_doh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block and dO block
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    do_ptrs = DO + do_off + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    do_block = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Load λ and D
    lam_ptrs = Lambda + off_hz * N_CTX + offs_m
    d_ptrs = D_sum + off_hz * N_CTX + offs_m
    lam_row = tl.load(lam_ptrs, mask=offs_m < N_CTX, other=0.0)
    d_row = tl.load(d_ptrs, mask=offs_m < N_CTX, other=1.0)

    # Accumulate Σ_j dP_ij · r_ij  (sweep over all K/V tiles)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v_block = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # Recompute attention helpers
        _weights, r, _qk = _stieltjes_score_helpers(
            q_block, k_block, lam_row, sm_scale, offs_m, offs_n,
            sq, EPS, CAUSAL,
        )

        # dP tile = dO @ V^T : [BLOCK_M, BLOCK_N]
        dP_tile = tl.dot(do_block, tl.trans(v_block))

        # Accumulate dP * r
        acc += tl.sum(dP_tile * r, axis=1)

    # δ_i = acc / D_i
    delta = acc / tl.maximum(d_row, EPS)

    delta_ptrs = Delta + off_hz * N_CTX + offs_m
    tl.store(delta_ptrs, delta, mask=offs_m < N_CTX)


@triton.jit
def _stieltjes_bwd_dkdv(
    Q, K, V, DO,
    Lambda, Delta,
    DK, DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    sm_scale, N_CTX,
    sq: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Compute dK and dV by iterating over Q blocks for a fixed K/V block."""
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    H_eff = stride_qz // stride_qh
    off_z = off_hz // H_eff
    off_h = off_hz % H_eff

    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    q_off = off_z * stride_qz + off_h * stride_qh
    do_off = off_z * stride_doz + off_h * stride_doh

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load K and V blocks (stay in SRAM for the entire inner loop)
    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    v_block = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Determine loop bounds (for causal, only Q rows >= K rows contribute)
    lo = 0
    if CAUSAL:
        lo = start_n * BLOCK_N

    for start_m in tl.range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        # Load Q, dO, delta, lambda for this Q block
        q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO + do_off + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
        q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        do_block = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        lam_row = tl.load(Lambda + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        delta_row = tl.load(Delta + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

        # Recompute P and r
        weights, r, _qk = _stieltjes_score_helpers(
            q_block, k_block, lam_row, sm_scale, offs_m, offs_n,
            sq, EPS, CAUSAL,
        )

        # dP = dO @ V^T : [BLOCK_M, BLOCK_N]
        dP_tile = tl.dot(do_block, tl.trans(v_block))

        # dS = sq * r * (dP - delta)
        dS = (sq * r * (dP_tile - delta_row[:, None])).to(q_block.dtype)

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            dS = tl.where(causal_mask, dS, 0.0)

        # dK += dS^T @ Q * sm_scale
        dk += tl.dot(tl.trans(dS), q_block) * sm_scale

        # dV += P^T @ dO
        dv += tl.dot(tl.trans(weights.to(do_block.dtype)), do_block)

    # Store dK, dV
    dk_off = off_z * stride_dkz + off_h * stride_dkh
    dv_off = off_z * stride_dvz + off_h * stride_dvh
    dk_ptrs = DK + dk_off + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    dv_ptrs = DV + dv_off + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
    tl.store(dk_ptrs, dk.to(k_block.dtype), mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptrs, dv.to(v_block.dtype), mask=offs_n[:, None] < N_CTX)


@triton.jit
def _stieltjes_bwd_dq(
    Q, K, V, DO,
    Lambda, Delta,
    DQ,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    sm_scale, N_CTX,
    sq: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Compute dQ by iterating over K/V blocks for a fixed Q block."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    H_eff = stride_qz // stride_qh
    off_z = off_hz // H_eff
    off_h = off_hz % H_eff

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    do_off = off_z * stride_doz + off_h * stride_doh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q, dO, delta, lambda (stay in SRAM)
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    do_ptrs = DO + do_off + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    do_block = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    lam_row = tl.load(Lambda + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    delta_row = tl.load(Delta + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Determine loop bounds (for causal, only K cols <= Q rows contribute)
    hi = N_CTX
    if CAUSAL:
        hi = (start_m + 1) * BLOCK_M

    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v_block = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # Recompute helpers
        _weights, r, _qk = _stieltjes_score_helpers(
            q_block, k_block, lam_row, sm_scale, offs_m, offs_n,
            sq, EPS, CAUSAL,
        )

        # dP = dO @ V^T : [BLOCK_M, BLOCK_N]
        dP_tile = tl.dot(do_block, tl.trans(v_block))

        # dS = sq * r * (dP - delta)
        dS = (sq * r * (dP_tile - delta_row[:, None])).to(q_block.dtype)

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            dS = tl.where(causal_mask, dS, 0.0)

        # dQ += dS @ K * sm_scale
        dq += tl.dot(dS, k_block) * sm_scale

    # Store dQ
    dq_off = off_z * stride_dqz + off_h * stride_dqh
    dq_ptrs = DQ + dq_off + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
    tl.store(dq_ptrs, dq.to(q_block.dtype), mask=offs_m[:, None] < N_CTX)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------

class StieltjesAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, stieltjes_q=1.0, num_iter=3):
        B, H, N, D = q.shape
        assert k.shape == v.shape == (B, H, N, D)
        assert D in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        lam = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
        d_sum = torch.empty((B * H, N), device=q.device, dtype=torch.float32)

        # Select block sizes based on head dimension
        if D <= 64:
            BLOCK_M, BLOCK_N = 128, 64
        else:
            BLOCK_M, BLOCK_N = 64, 64

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
            LAMBDA_INIT=float(N) ** (1.0 / stieltjes_q),
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, lam, d_sum)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.stieltjes_q = stieltjes_q
        ctx.num_iter = num_iter
        return o

    @staticmethod
    def backward(ctx, do):
        """
        Triton backward for Stieltjes attention (3 kernels, flash-style).

        Given P_ij = (λ_i - s_ij)^{-q}, the score gradient is:
            dS_ij = q · r_ij · (dP_ij − δ_i)
        where r_ij = (λ_i - s_ij)^{-q-1},  δ_i = (Σ_k dP_ik · r_ik) / D_i.

        All kernels recompute scores on-the-fly to avoid O(N²) storage.
        """
        q, k, v, lam, d_sum = ctx.saved_tensors
        sq = ctx.stieltjes_q
        sm_scale = ctx.sm_scale
        causal = ctx.causal

        B, H, N, D = q.shape
        BH = B * H
        if D <= 64:
            BLOCK_M, BLOCK_N = 128, 64
        else:
            BLOCK_M, BLOCK_N = 64, 64

        do = do.contiguous()

        # Pass all 4 strides (B, H, N, D) for correct non-contiguous support
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        do_strides = (do.stride(0), do.stride(1), do.stride(2), do.stride(3))

        common_args = dict(
            sm_scale=sm_scale, N_CTX=N,
            sq=sq, EPS=1e-6,
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=causal,
        )

        # --- Kernel 1: Compute delta ---
        delta = torch.empty((BH, N), device=q.device, dtype=torch.float32)
        grid_m = (triton.cdiv(N, BLOCK_M), BH)

        _stieltjes_bwd_delta[grid_m](
            q, k, v, do,
            lam, d_sum, delta,
            *q_strides, *k_strides, *v_strides, *do_strides,
            **common_args,
        )

        # --- Kernel 2: Compute dK, dV ---
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        grid_n = (triton.cdiv(N, BLOCK_N), BH)

        _stieltjes_bwd_dkdv[grid_n](
            q, k, v, do,
            lam, delta,
            dk, dv,
            *q_strides, *k_strides, *v_strides, *do_strides,
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            **common_args,
        )

        # --- Kernel 3: Compute dQ ---
        dq = torch.empty_like(q)

        _stieltjes_bwd_dq[grid_m](
            q, k, v, do,
            lam, delta,
            dq,
            *q_strides, *k_strides, *v_strides, *do_strides,
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            **common_args,
        )

        return dq, dk, dv, None, None, None, None


def stieltjes_attention(q, k, v, causal=False, sm_scale=None, stieltjes_q=1.0, num_iter=3):
    """
    Stieltjes flash attention.

    Args:
        q, k, v: (B, H, N, D) — query, key, value tensors
        causal: whether to apply causal masking
        sm_scale: attention scale factor (default: 1/sqrt(D))
        stieltjes_q: order of the Stieltjes transform (default 1.0)
        num_iter: Newton-Raphson iterations (default 3)
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return StieltjesAttention.apply(q, k, v, causal, sm_scale, stieltjes_q, num_iter)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_forward_correctness():
    torch.manual_seed(42)
    DEVICE = torch.device('cuda', triton.runtime.driver.active.get_current_device())

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
        # Larger N to catch tiling bugs across multiple blocks
        (1, 2, 512, 64,  False, 1.0),
        (1, 2, 512, 64,  True,  1.0),
        (1, 1, 1024, 64, False, 1.0),
        (1, 1, 1024, 64, True,  2.0),
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
    DEVICE = torch.device('cuda', triton.runtime.driver.active.get_current_device())

    configs = [
        # (B, H, N, D, causal, q)
        (1, 1, 64,  64, False, 1.0),
        (1, 1, 64,  64, True,  1.0),
        (2, 2, 128, 64, False, 1.0),
        (2, 2, 128, 64, True,  1.0),
        (1, 1, 128, 128, False, 1.0),
        (1, 2, 128, 64, False, 2.0),
        # Larger N to catch tiling bugs across multiple blocks
        (1, 2, 512, 64, False, 1.0),
        (1, 2, 512, 64, True,  1.0),
        (1, 1, 1024, 64, False, 1.0),
    ]

    print("\nBackward correctness tests (Triton bwd vs PyTorch autograd reference)")
    print("-" * 70)
    all_passed = True

    for B, H, N, D, causal, sq in configs:
        sm_scale = 1.0 / (D ** 0.5)

        # Reference: float32 PyTorch autograd on dense reference
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

        # Triton forward + Triton backward (all fp16)
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

        passed = max(dq_err, dk_err, dv_err) < 0.15  # fp16 + iterative solver tolerance
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

    DEVICE = torch.device('cuda', triton.runtime.driver.active.get_current_device())

    bench_configs = []
    for B, H, D in [(4, 8, 64), (4, 8, 128)]:
        for mode in ["fwd", "bwd"]:
            bench_configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(7, 13)],
                    line_arg="provider",
                    line_vals=["stieltjes-triton", "stieltjes-torch", "softmax-torch"],
                    line_names=["Stieltjes (Triton)", "Stieltjes (PyTorch)", "Softmax (PyTorch)"],
                    styles=[("red", "-"), ("blue", "--"), ("green", ":")],
                    ylabel="TFLOPS",
                    plot_name=f"stieltjes-flash-attention-{mode}-B{B}-H{H}-D{D}",
                    args={"B": B, "H": H, "D": D, "mode": mode},
                ))

    @triton.testing.perf_report(bench_configs)
    def bench_fn(B, H, N_CTX, D, mode, provider, device=DEVICE):
        sm_scale = 1.0 / (D ** 0.5)

        if provider == "stieltjes-triton":
            dtype = torch.float16
            q = torch.randn(B, H, N_CTX, D, dtype=dtype, device=device, requires_grad=True)
            k = torch.randn(B, H, N_CTX, D, dtype=dtype, device=device, requires_grad=True)
            v = torch.randn(B, H, N_CTX, D, dtype=dtype, device=device, requires_grad=True)
            fn = lambda: stieltjes_attention(q, k, v, sm_scale=sm_scale)
        elif provider == "stieltjes-torch":
            # Use float32 inputs directly so backward flows through
            q = torch.randn(B, H, N_CTX, D, dtype=torch.float32, device=device, requires_grad=True)
            k = torch.randn(B, H, N_CTX, D, dtype=torch.float32, device=device, requires_grad=True)
            v = torch.randn(B, H, N_CTX, D, dtype=torch.float32, device=device, requires_grad=True)
            fn = lambda: stieltjes_attention_ref(q, k, v, sm_scale)
        elif provider == "softmax-torch":
            # Use float32 inputs directly so backward flows through
            q = torch.randn(B, H, N_CTX, D, dtype=torch.float32, device=device, requires_grad=True)
            k = torch.randn(B, H, N_CTX, D, dtype=torch.float32, device=device, requires_grad=True)
            v = torch.randn(B, H, N_CTX, D, dtype=torch.float32, device=device, requires_grad=True)
            def fn():
                s = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
                p = torch.softmax(s, dim=-1)
                return torch.matmul(p, v)

        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)

        ms = triton.testing.do_bench(fn)
        # 2 matmuls: QK^T and PV, each 2*B*H*N*N*D flops
        flops = 2 * 2.0 * B * H * N_CTX * N_CTX * D
        if mode == "bwd":
            flops *= 2.5  # backward is ~2.5x the flops of forward
        return flops * 1e-12 / (ms * 1e-3)

    bench_fn.run(save_path=".", print_data=True)


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    fwd_ok = test_forward_correctness()
    bwd_ok = test_backward_correctness()
    if fwd_ok and bwd_ok:
        print("\n\nRunning benchmarks...\n")
        benchmark()
