"""
Long-context scaling benchmark: throughput and peak memory vs sequence length.

Compares four providers across N = 1024 .. 131072:
  - stieltjes_triton : our Triton flash kernel  (O(N·d) memory)
  - flash_sdpa       : torch SDPA / cuDNN flash  (O(N·d) memory)
  - stieltjes_ref    : PyTorch reference Stieltjes (O(N²) memory — OOMs early)
  - naive_softmax    : materialised softmax       (O(N²) memory — OOMs early)

Output CSV: $RESULTS_DIR/bench_longctx_scaling_<gpu>.csv
Columns: provider, N, D, causal, q, fwd_ms, fwd_tflops, fwd_peak_mb, oom
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch
import triton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref  # noqa: E402

DEVICE = torch.device("cuda")
DTYPE  = torch.bfloat16

# Fixed: B=1 so we can reach large N without OOM on the QKV tensors themselves.
B = 1
H = 8
D = 64         # head dim
CAUSAL = True  # causal is the realistic training setting
Q_STJ  = 4.0  # representative Stieltjes order

N_VALS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

WARMUP = 10
REP    = 50


def flops_fwd(B, H, N, D, causal):
    ops = 2 * 2.0 * B * H * N * N * D
    return ops * 0.5 if causal else ops


def measure(fn, N) -> tuple[float, float]:
    """Returns (ms, peak_mem_mb). Measures peak memory on first call, then benches."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    try:
        fn()  # one warmup call for memory measurement
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), float("nan")

    try:
        ms = triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), peak_mb

    return ms, peak_mb


def make_tensors(N):
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    return q, k, v


def bench_stieltjes_triton(N):
    q, k, v = make_tensors(N)
    sm_scale = D ** -0.5
    fn = lambda: stieltjes_attention(q, k, v, causal=CAUSAL, sm_scale=sm_scale,
                                     stieltjes_q=Q_STJ, num_iter=3)
    return measure(fn, N)


def bench_flash_sdpa(N):
    q, k, v = make_tensors(N)
    fn = lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=CAUSAL)
    return measure(fn, N)


def bench_stieltjes_ref(N):
    q, k, v = make_tensors(N)
    sm_scale = D ** -0.5
    fn = lambda: stieltjes_attention_ref(q, k, v, sm_scale=sm_scale, causal=CAUSAL,
                                         stieltjes_q=Q_STJ, num_iter=3)
    return measure(fn, N)


def bench_naive_softmax(N):
    q, k, v = make_tensors(N)
    sm_scale = D ** -0.5

    def fn():
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        if CAUSAL:
            mask = torch.triu(torch.ones(N, N, device=DEVICE, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(DTYPE)
        return torch.matmul(attn, v)

    return measure(fn, N)


PROVIDERS = [
    ("stieltjes_triton", bench_stieltjes_triton),
    ("flash_sdpa",       bench_flash_sdpa),
    ("stieltjes_ref",    bench_stieltjes_ref),
    ("naive_softmax",    bench_naive_softmax),
]


def gpu_tag():
    name = torch.cuda.get_device_name(0).lower()
    if "h100" in name:
        return "h100"
    if "a100" in name:
        return "a100"
    return name.replace(" ", "_")[:20]


def main():
    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = gpu_tag()
    out_path = os.path.join(results_dir, f"bench_longctx_scaling_{tag}.csv")

    fieldnames = ["provider", "N", "D", "H", "B", "causal", "q",
                  "fwd_ms", "fwd_tflops", "fwd_peak_mb", "oom"]

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"B={B} H={H} D={D} causal={CAUSAL} q_stj={Q_STJ}", flush=True)
    print(f"Output: {out_path}", flush=True)

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for provider, bench_fn in PROVIDERS:
            for N in N_VALS:
                print(f"  {provider:22s}  N={N:7d} ...", end=" ", flush=True)
                ms, peak_mb = bench_fn(N)
                oom = (ms != ms)  # nan check

                if oom:
                    tflops = float("nan")
                    print("OOM", flush=True)
                else:
                    tf = flops_fwd(B, H, N, D, CAUSAL) * 1e-12 / (ms * 1e-3)
                    tflops = tf
                    print(f"{ms:.2f} ms  {tf:.2f} TFLOPS  {peak_mb:.0f} MB", flush=True)

                writer.writerow(dict(
                    provider=provider, N=N, D=D, H=H, B=B,
                    causal=CAUSAL, q=Q_STJ,
                    fwd_ms=f"{ms:.4f}" if not oom else "OOM",
                    fwd_tflops=f"{tflops:.4f}" if not oom else "OOM",
                    fwd_peak_mb=f"{peak_mb:.1f}" if not oom else "OOM",
                    oom=int(oom),
                ))
                fh.flush()

    print(f"\nDone. {out_path}", flush=True)


if __name__ == "__main__":
    main()
