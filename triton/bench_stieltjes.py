import torch
import triton
from stieltjes import (
    stieltjes_torch, stieltjes,
    stieltjes_bsearch_torch, stieltjes_bsearch,
    DEVICE,
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(8, 18)],  # 256 to 131072
        line_arg='provider',
        line_vals=['triton', 'torch', 'triton_bsearch', 'torch_bsearch', 'softmax'],
        line_names=['Triton NR', 'PyTorch NR', 'Triton BSearch', 'PyTorch BSearch', 'Softmax'],
        styles=[('blue', '-'), ('green', '-'), ('blue', '--'), ('green', '--'), ('red', '-')],
        ylabel='GB/s',
        plot_name='stieltjes-performance',
        args={'M': 4096},
    ))
def benchmark(M, N, provider):
    # 
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: stieltjes_torch(x))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: stieltjes(x))
    elif provider == 'torch_bsearch':
        ms = triton.testing.do_bench(lambda: stieltjes_bsearch_torch(x))
    elif provider == 'triton_bsearch':
        ms = triton.testing.do_bench(lambda: stieltjes_bsearch(x))
    elif provider == 'softmax':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == '__main__':
    print(f"Device: {DEVICE}\n")
    benchmark.run(save_path='.', print_data=True)
