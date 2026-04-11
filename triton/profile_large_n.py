"""
Profile Stieltjes Flash Attention vs Softmax at Large N
=======================================================

Benchmarks forward and backward TFLOPS for N = {8192, 16384, 32768}.
Results saved to $RESULTS_DIR/profile_large_n.csv (default: current dir).
"""

import os
import csv
import math
import torch
import triton

from stieltjes_flash_attn import stieltjes_attention

DEVICE = torch.device("cuda")

# Benchmark configuration
N_VALS = [8192, 16384, 32768]
B = 1
H = 8
D = 64

RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
CSV_PATH = os.path.join(RESULTS_DIR, "profile_large_n.csv")

CSV_COLUMNS = ["N", "mode", "provider", "tflops", "ms", "B", "H", "D"]


def softmax_forward(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def make_tensors(B, H, N, D, requires_grad=False):
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16,
                    requires_grad=requires_grad)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16,
                    requires_grad=requires_grad)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16,
                    requires_grad=requires_grad)
    return q, k, v


def compute_tflops(B, H, N, D, mode, ms):
    flops = 2 * 2.0 * B * H * N * N * D
    if mode == "bwd":
        flops *= 2.5
    return flops * 1e-12 / (ms * 1e-3)


def bench_forward(fn):
    try:
        ms = triton.testing.do_bench(fn)
        return ms
    except torch.cuda.OutOfMemoryError:
        return None


def bench_backward(fwd_fn):
    """Run one forward pass to get output, then benchmark backward."""
    try:
        # Warm-up forward to get dout shape
        out = fwd_fn()
        dout = torch.randn_like(out)

        def bwd():
            o = fwd_fn()
            o.backward(dout)

        ms = triton.testing.do_bench(bwd)
        return ms
    except torch.cuda.OutOfMemoryError:
        return None


def run_benchmark(N, mode, provider):
    scale = 1.0 / math.sqrt(D)
    needs_grad = (mode == "bwd")

    try:
        q, k, v = make_tensors(B, H, N, D, requires_grad=needs_grad)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM allocating tensors for N={N}, mode={mode}, provider={provider}")
        return None, None

    if provider == "stieltjes-triton":
        fn = lambda: stieltjes_attention(q, k, v, sm_scale=scale)
    elif provider == "softmax-torch":
        fn = lambda: softmax_forward(q, k, v, scale)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if mode == "fwd":
        ms = bench_forward(fn)
    else:
        ms = bench_backward(fn)

    if ms is None:
        return None, None

    tflops = compute_tflops(B, H, N, D, mode, ms)
    return tflops, ms


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    providers = ["stieltjes-triton", "softmax-torch"]
    modes = ["fwd", "bwd"]

    print(f"Profiling Stieltjes vs Softmax at large N")
    print(f"Config: B={B}, H={H}, D={D}")
    print(f"Output: {CSV_PATH}")
    print()

    for N in N_VALS:
        for mode in modes:
            for provider in providers:
                print(f"  N={N:6d}  mode={mode}  provider={provider} ...", end=" ", flush=True)

                torch.cuda.empty_cache()
                tflops, ms = run_benchmark(N, mode, provider)

                if tflops is None:
                    print("OOM")
                    rows.append({
                        "N": N, "mode": mode, "provider": provider,
                        "tflops": "OOM", "ms": "OOM",
                        "B": B, "H": H, "D": D,
                    })
                else:
                    print(f"{tflops:.2f} TFLOPS  ({ms:.3f} ms)")
                    rows.append({
                        "N": N, "mode": mode, "provider": provider,
                        "tflops": f"{tflops:.4f}", "ms": f"{ms:.4f}",
                        "B": B, "H": H, "D": D,
                    })

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {CSV_PATH}")


if __name__ == "__main__":
    main()
