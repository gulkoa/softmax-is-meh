import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
if __name__ == "__main__":
    print(f"device is {DEVICE}")

def naive_softmax(x):
  x_max = x.max(dim=1)[0]
  z = x - x_max[:, None]
  e_to_z = torch.exp(z)
  sum_e_to_z = e_to_z.sum(dim=1)

  return e_to_z / sum_e_to_z[:, None]

def pytorch_softmax(x):
  return torch.softmax(x, dim=1)

NEGATIVE_INF = tl.constexpr(-float('inf'))

@triton.jit
def softmax_kernel(x_ptr, output_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
      row_start_ptr = x_ptr + row_idx * input_row_stride
      col_offsets = tl.arange(0, BLOCK_SIZE)
      input_ptrs = row_start_ptr + col_offsets
      
      mask = col_offsets < n_cols
      row = tl.load(input_ptrs, mask=mask, other=NEGATIVE_INF)
      row_minus_max = row - tl.max(row, axis=0)
      exp_row = tl.exp(row_minus_max)
      sum_exp_row = tl.sum(exp_row, axis=0)
      softmax_row = exp_row / sum_exp_row
      
      output_row_start_ptr = output_ptr + row_idx * output_row_stride
      output_ptrs = output_row_start_ptr + col_offsets
      tl.store(output_ptrs, softmax_row, mask=mask)

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # smallest power of 2 greater than or equal to n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200_000 else 2

    y = torch.empty_like(x)

    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1, 1))
    kernel._init_handles()

    size_smem = kernel.metadata.shared
    
    occupancy = NUM_REGS // (n_rows * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))

def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)