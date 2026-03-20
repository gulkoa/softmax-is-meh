import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def stieltjes_torch(x: torch.Tensor, q: float = 1.0, num_iter: int = 3, eps: float = 1e-9) -> torch.Tensor:
    """
    Stieltjes transform along dim=-1, matching the Newton-Raphson variant.
    x: (M, N)
    q: order of the transform
    num_iter: number of iterations for Newton-Raphson
    eps: epsilon for numerical stability
    """
    x_max = x.max(dim=-1, keepdim=True).values # len(x_max) = batch
    x_i = x - x_max # center around 0, max = 0

    n = x.shape[-1]
    lambd = torch.full_like(x_max, n ** (1.0 / q)) #initial guess = n^(1/q)

    for _ in range(num_iter):
        diff = (lambd - x_i).clamp(min=eps) # clamp to avoid division by zero
        f_val  = torch.sum(torch.pow(diff, -q), dim=-1, keepdim=True) - 1.0 # f(λ) = Σ (λ - x_i)^(-q) - 1
        f_deriv = -q * torch.sum(torch.pow(diff, -q - 1.0), dim=-1, keepdim=True) # f'(λ) = -q Σ (λ - x_i)^(-q-1)
        lambd = lambd - (f_val / f_deriv) # Newton-Raphson update

    return torch.pow((lambd - x_i).clamp(min=eps), -q) # 1 / (λ - x_i)^q

# triton kernel - blocked / tiled (flash-attention style)
# instead of loading the full row into one BLOCK_SIZE vector, we tile over columns in three passes so that BLOCK_SIZE can be much smaller than n_cols
# passes per row:
#   1. Find row max (1 tiled pass)
#   2. Newton-Raphson for λ (num_iter tiled passes)
#   3. Compute & store output (1 tiled pass)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['n_cols'],
)
@triton.jit
def stieltjes_kernel(
    x_ptr,
    output_ptr,
    stride_x,
    stride_out,
    n_cols,
    init_lambda,
    q: tl.constexpr,
    num_iter: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row = x_ptr + row_idx * stride_x
    out_row = output_ptr + row_idx * stride_out

    # pass 1: row maximum (tiled)
    row_max = -float('inf')
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # pass 2: Newton-Raphson to find λ (one full tiled sweep per iteration)
    # after shifting by row_max, max(x_i) = 0, so λ must be > 0.
    # initial guess n^(1/q) is exact when all x_i are equal.
    # NR's quadratic convergence then needs ~3 iterations for float32.
    lambd = init_lambda
    for _ in tl.static_range(num_iter):
        f_val = 0.0
        f_deriv = 0.0
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            # Masked positions load -inf → diff = lambd-(-inf) = +inf
            # → 1/inf = 0, pow(inf,-q) = 0, so they contribute nothing to sums
            x = tl.load(x_row + offs, mask=offs < n_cols, other=-float('inf'))
            diff = tl.maximum(lambd - (x - row_max), eps)
            if q == 1.0:
                inv_q = 1.0 / diff
                inv_q1 = inv_q * inv_q
            else:
                inv_q = diff ** (-q)
                inv_q1 = diff ** (-q - 1.0)
            f_val += tl.sum(inv_q, axis=0)
            f_deriv += tl.sum(inv_q1, axis=0)
        # f(λ) = Σ (λ - x_i)^{-q} - 1,  f'(λ) = -q Σ (λ - x_i)^{-q-1}
        f_val -= 1.0
        f_deriv *= -q
        lambd -= f_val / f_deriv

    # pass 3: compute and store output (tiled)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=-float('inf'))
        out_diff = tl.maximum(lambd - (x - row_max), eps)
        if q == 1.0:
            out = 1.0 / out_diff
        else:
            out = out_diff ** (-q)
        tl.store(out_row + offs, out, mask=mask)

def stieltjes(x: torch.Tensor, q: float = 1.0, num_iter: int = 3, eps: float = 1e-9) -> torch.Tensor:
    n_rows, n_cols = x.shape
    init_lambda = float(n_cols) ** (1.0 / q)
    y = torch.empty_like(x)
    stieltjes_kernel[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols, init_lambda,
        q=q, num_iter=num_iter, eps=eps,
    )
    return y


# Binary-search variants
# Instead of Newton-Raphson, bracket the root of f(λ)=Σ(λ-x_i)^{-q}-1=0
# in [eps, n^{1/q}] and bisect.  More iterations than NR (≈32 vs 3) but
# each step is branch-free and needs no derivative evaluation.

def stieltjes_bsearch_torch(x: torch.Tensor, q: float = 1.0, num_iter: int = 5, eps: float = 1e-9) -> torch.Tensor:
    """Stieltjes transform along dim=-1, using binary search for λ."""
    x_max = x.max(dim=-1, keepdim=True).values
    x_i = x - x_max

    n = x.shape[-1]
    lo = torch.full_like(x_max, eps)
    hi = torch.full_like(x_max, float(n) ** (1.0 / q))

    for _ in range(num_iter):
        mid = (lo + hi) * 0.5
        diff = (mid - x_i).clamp(min=eps)
        f_val = torch.sum(torch.pow(diff, -q), dim=-1, keepdim=True)
        # f is decreasing: f > 1 ⇒ λ too small, f ≤ 1 ⇒ λ too large
        lo = torch.where(f_val > 1.0, mid, lo)
        hi = torch.where(f_val <= 1.0, mid, hi)

    lambd = (lo + hi) * 0.5
    return torch.pow((lambd - x_i).clamp(min=eps), -q)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['n_cols'],
)
@triton.jit
def stieltjes_bsearch_kernel(
    x_ptr,
    output_ptr,
    stride_x,
    stride_out,
    n_cols,
    init_lambda,
    q: tl.constexpr,
    num_iter: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row = x_ptr + row_idx * stride_x
    out_row = output_ptr + row_idx * stride_out

    # Pass 1: row maximum (tiled)
    row_max = -float('inf')
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Pass 2: binary search for λ
    # f(λ) = Σ (λ - x_i)^{-q} is monotonically decreasing for λ > max(x_i).
    # Bracket the root of f(λ) - 1 = 0 in [eps, n^{1/q}].
    lo = eps
    hi = init_lambda
    for _ in tl.static_range(num_iter):
        mid = (lo + hi) * 0.5
        f_val = 0.0
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(x_row + offs, mask=offs < n_cols, other=-float('inf'))
            diff = tl.maximum(mid - (x - row_max), eps)
            if q == 1.0:
                inv_q = 1.0 / diff
            else:
                inv_q = diff ** (-q)
            f_val += tl.sum(inv_q, axis=0)
        # f_val > 1 → λ too small → raise lower bound
        # f_val ≤ 1 → λ too large → lower upper bound
        if f_val > 1.0:
            lo = mid
        else:
            hi = mid

    lambd = (lo + hi) * 0.5

    # Pass 3: compute and store output (tiled)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=-float('inf'))
        out_diff = tl.maximum(lambd - (x - row_max), eps)
        if q == 1.0:
            out = 1.0 / out_diff
        else:
            out = out_diff ** (-q)
        tl.store(out_row + offs, out, mask=mask)


def stieltjes_bsearch(x: torch.Tensor, q: float = 1.0, num_iter: int = 5, eps: float = 1e-9) -> torch.Tensor:
    n_rows, n_cols = x.shape
    init_lambda = float(n_cols) ** (1.0 / q)
    y = torch.empty_like(x)
    stieltjes_bsearch_kernel[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols, init_lambda,
        q=q, num_iter=num_iter, eps=eps,
    )
    return y


# correctness check

def test_correctness():
    torch.manual_seed(42)
    for n_cols in [16, 32, 64, 128, 256, 1024, 4096, 8192]:
        x = torch.randn(64, n_cols, device=DEVICE, dtype=torch.float32)
        y_ref       = stieltjes_torch(x)
        y_triton    = stieltjes(x)

        ref_sum = y_ref.sum(dim=-1)
        tri_sum = y_triton.sum(dim=-1)

        max_diff    = (y_triton - y_ref).abs().max().item()
        ref_sum_err = (ref_sum - 1.0).abs().max().item()
        tri_sum_err = (tri_sum - 1.0).abs().max().item()
        print(f"  n_cols={n_cols:5d}  [NR]      max_diff={max_diff:.2e}  "
              f"ref_sum={ref_sum.mean():.6f}  tri_sum={tri_sum.mean():.6f}")

        assert max_diff < 1e-4, f"NR mismatch for n_cols={n_cols}: max diff = {max_diff}"
        assert ref_sum_err < 1e-4, f"NR ref rows don't sum to 1 for n_cols={n_cols}: max err = {ref_sum_err}"
        assert tri_sum_err < 1e-4, f"NR triton rows don't sum to 1 for n_cols={n_cols}: max err = {tri_sum_err}"

        # binary-search variants
        y_bs_ref    = stieltjes_bsearch_torch(x)
        y_bs_triton = stieltjes_bsearch(x)

        bs_ref_sum = y_bs_ref.sum(dim=-1)
        bs_tri_sum = y_bs_triton.sum(dim=-1)

        bs_ref_diff = (y_bs_ref - y_ref).abs().max().item()
        bs_tri_diff = (y_bs_triton - y_ref).abs().max().item()
        bs_ref_err  = (bs_ref_sum - 1.0).abs().max().item()
        bs_tri_err  = (bs_tri_sum - 1.0).abs().max().item()
        print(f"  n_cols={n_cols:5d}  [bsearch] ref_diff={bs_ref_diff:.2e}  tri_diff={bs_tri_diff:.2e}  "
              f"ref_sum={bs_ref_sum.mean():.6f}  tri_sum={bs_tri_sum.mean():.6f}")

        assert bs_ref_diff < 1e-4, f"BSearch torch mismatch for n_cols={n_cols}: diff = {bs_ref_diff}"
        assert bs_tri_diff < 1e-4, f"BSearch triton mismatch for n_cols={n_cols}: diff = {bs_tri_diff}"
        assert bs_ref_err < 1e-4, f"BSearch torch rows don't sum to 1 for n_cols={n_cols}: err = {bs_ref_err}"
        assert bs_tri_err < 1e-4, f"BSearch triton rows don't sum to 1 for n_cols={n_cols}: err = {bs_tri_err}"

    print("All correctness checks passed.\n")


if __name__ == '__main__':
    print(f"Device: {DEVICE}\n")
    test_correctness()
