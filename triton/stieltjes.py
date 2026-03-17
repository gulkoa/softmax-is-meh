import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def stieltjes_torch(x: torch.Tensor, q: float = 1.0, num_iter: int = 3, eps: float = 1e-9) -> torch.Tensor:
    """Stieltjes transform along dim=-1, matching the Newton-Raphson variant."""
    x_max = x.max(dim=-1, keepdim=True).values
    x_i = x - x_max

    n = x.shape[-1]
    lambd = torch.full_like(x_max, n ** (1.0 / q))

    for _ in range(num_iter):
        diff = (lambd - x_i).clamp(min=eps)
        f_val  = torch.sum(torch.pow(diff, -q),       dim=-1, keepdim=True) - 1.0
        f_deriv = -q * torch.sum(torch.pow(diff, -q - 1.0), dim=-1, keepdim=True)
        lambd = lambd - (f_val / f_deriv)

    return torch.pow((lambd - x_i).clamp(min=eps), -q)

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

    # Pass 1: row maximum (tiled)
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


# correctness check

def test_correctness():
    torch.manual_seed(42)
    for n_cols in [37, 128, 256, 1024, 4096, 8192]:
        x = torch.randn(64, n_cols, device=DEVICE, dtype=torch.float32)
        y_ref    = stieltjes_torch(x)
        y_triton = stieltjes(x)

        ref_sum = y_ref.sum(dim=-1)
        tri_sum = y_triton.sum(dim=-1)

        max_diff    = (y_triton - y_ref).abs().max().item()
        ref_sum_err = (ref_sum - 1.0).abs().max().item()
        tri_sum_err = (tri_sum - 1.0).abs().max().item()
        print(f"  n_cols={n_cols:5d}  max_abs_diff={max_diff:.2e}  "
              f"ref_row_sum={ref_sum.mean():.6f}  triton_row_sum={tri_sum.mean():.6f}")

        assert max_diff < 1e-4, f"Mismatch for n_cols={n_cols}: max diff = {max_diff}"
        assert ref_sum_err < 1e-4, f"Ref rows don't sum to 1 for n_cols={n_cols}: max err = {ref_sum_err}"
        assert tri_sum_err < 1e-4, f"Triton rows don't sum to 1 for n_cols={n_cols}: max err = {tri_sum_err}"

    print("All correctness checks passed.\n")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton Stieltjes', 'PyTorch Stieltjes'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='stieltjes-performance',
        args={'M': 4096},
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: stieltjes_torch(x))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: stieltjes(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == '__main__':
    print(f"Device: {DEVICE}\n")
    test_correctness()
    benchmark.run(show_plots=True, print_data=True)
