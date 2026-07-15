"""
FLASHn long-N throughput microbench: BH x D x N x num_iter vs SDPA.

Motivation (task: MQMTAR eval forensics): at BH=1024 (bs64 x 16h), D=32,
N=32k-65k, causal bf16, the fused Stieltjes forward measured ~80x slower
than SDPA on identical geometry, while the sweep count (2 + num_iter = 10
N^2 passes vs SDPA's 1) explains only ~10x. Prior benches
(bench_flashn_vs_jack, 131k feasibility) used BH=64, D=64 and looked fine.

Matrix: BH in {64 (4x16), 1024 (64x16)} x D in {32, 64} x
        N in {4096, 16384, 65536} x num_iter in {1, 8}, causal, bf16,
        forward only (inference mode — matches the generation eval).
SDPA baseline per (BH, D, N). Prints a table + writes CSV to results/.

Usage (compute node): uv run --project softmax-is-meh/triton --no-sync \
    python softmax-is-meh/triton/bench_flashn_bh_scaling.py
"""
import csv
import itertools
import math
import os
import sys

import torch
import torch.nn.functional as F

# cuDNN SDPA builds no execution plan on this torch/H100 combo (also hit
# by MQMTAR job 12345902) — use flash/efficient backends
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402

OUT_CSV = ("/users/PAS2402/alexg/softmax/softmax-is-meh/results/"
           f"flashn_bh_scaling_{os.environ.get('SLURM_JOB_ID', 'local')}.csv")


def time_fn(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms


@torch.inference_mode()
def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    rows = []
    BH_CONFIGS = [(4, 16), (64, 16)]          # BH = 64, 1024
    for (B, H), D, N in itertools.product(
            BH_CONFIGS, [32, 64], [4096, 16384, 65536]):
        torch.manual_seed(0)
        try:
            q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
        except torch.OutOfMemoryError:
            print(f"BH={B*H} D={D} N={N}: alloc OOM, skipped", flush=True)
            continue
        sm = 1.0 / math.sqrt(D)

        ms_sdpa = time_fn(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True))
        row = {"BH": B * H, "D": D, "N": N, "sdpa_ms": round(ms_sdpa, 2)}

        for ni in [1, 8]:
            try:
                ms = time_fn(lambda: stieltjes_attention(
                    q, k, v, causal=True, sm_scale=sm, stieltjes_q=4.0,
                    num_iter=ni, normalize=True), warmup=1, iters=3)
            except torch.OutOfMemoryError:
                ms = float("nan")
            row[f"flashn_i{ni}_ms"] = round(ms, 2)
            row[f"ratio_i{ni}"] = round(ms / ms_sdpa, 1)

        # sweep-count model: expected ratio ~ (2 + num_iter) passes vs 1
        row["excess_i8"] = round(row["ratio_i8"] / 10.0, 1)
        rows.append(row)
        print(f"BH={B*H:4d} D={D:3d} N={N:6d}: sdpa {ms_sdpa:8.2f}ms | "
              f"flashn i1 {row['flashn_i1_ms']:9.2f}ms ({row['ratio_i1']:6.1f}x) | "
              f"i8 {row['flashn_i8_ms']:9.2f}ms ({row['ratio_i8']:6.1f}x, "
              f"excess {row['excess_i8']:4.1f}x)", flush=True)
        del q, k, v
        torch.cuda.empty_cache()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
