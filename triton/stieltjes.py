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
        lambd = torch.maximum(lambd - f_val / f_deriv, lambd * 0.5) # NR update; halving guard prevents overshoot

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
                log_diff = tl.log(diff)
                inv_q = tl.exp(log_diff * (-q))
                inv_q1 = tl.exp(log_diff * (-q - 1.0))
            f_val += tl.sum(inv_q, axis=0)
            f_deriv += tl.sum(inv_q1, axis=0)
        # f(λ) = Σ (λ - x_i)^{-q} - 1,  f'(λ) = -q Σ (λ - x_i)^{-q-1}
        f_val -= 1.0
        f_deriv *= -q
        lambd = tl.maximum(lambd - f_val / f_deriv, lambd * 0.5)

    # pass 3: compute and store output (tiled)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=-float('inf'))
        out_diff = tl.maximum(lambd - (x - row_max), eps)
        if q == 1.0:
            out = 1.0 / out_diff
        else:
            out = tl.exp(tl.log(out_diff) * (-q))
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
                inv_q = tl.exp(tl.log(diff) * (-q))
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
            out = tl.exp(tl.log(out_diff) * (-q))
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
    n_cols_list = [16, 32, 64, 128, 256, 1024, 4096, 8192]
    q_vals = [1.0, 2.0, 4.0]

    for q in q_vals:
        print(f"── q = {q} ──")
        for n_cols in n_cols_list:
            x = torch.randn(64, n_cols, device=DEVICE, dtype=torch.float32)

            # ground truth: float64 binary search with 100 iterations
            x64 = x.to(torch.float64)
            y_exact = stieltjes_bsearch_torch(x64, q=q, num_iter=100, eps=1e-15)
            exact_sum_err = (y_exact.sum(dim=-1) - 1.0).abs().max().item()
            y_exact_f32 = y_exact.to(torch.float32)

            # fast variants (all float32)
            y_nr_torch  = stieltjes_torch(x, q=q)
            y_nr_triton = stieltjes(x, q=q)
            y_bs_torch  = stieltjes_bsearch_torch(x, q=q)
            y_bs_triton = stieltjes_bsearch(x, q=q)

            nr_torch_err  = (y_nr_torch  - y_exact_f32).abs().max().item()
            nr_triton_err = (y_nr_triton - y_exact_f32).abs().max().item()
            bs_torch_err  = (y_bs_torch  - y_exact_f32).abs().max().item()
            bs_triton_err = (y_bs_triton - y_exact_f32).abs().max().item()

            nr_torch_sum  = (y_nr_torch.sum(dim=-1)  - 1.0).abs().max().item()
            nr_triton_sum = (y_nr_triton.sum(dim=-1) - 1.0).abs().max().item()
            bs_torch_sum  = (y_bs_torch.sum(dim=-1)  - 1.0).abs().max().item()
            bs_triton_sum = (y_bs_triton.sum(dim=-1) - 1.0).abs().max().item()

            print(f"  n={n_cols:5d}  exact_sum_err={exact_sum_err:.2e}")
            print(f"    NR  torch={nr_torch_err:.2e}  triton={nr_triton_err:.2e}  "
                  f"sum_err: torch={nr_torch_sum:.2e}  triton={nr_triton_sum:.2e}")
            print(f"    BS  torch={bs_torch_err:.2e}  triton={bs_triton_err:.2e}  "
                  f"sum_err: torch={bs_torch_sum:.2e}  triton={bs_triton_sum:.2e}")

            assert exact_sum_err < 1e-12, \
                f"Ground truth not converged at q={q}, n={n_cols}: sum_err={exact_sum_err}"

    print("\nAll correctness checks passed.")


if __name__ == '__main__':
    print(f"Device: {DEVICE}\n")
    test_correctness()
