"""
Final benchmark: Stieltjes flash attention vs softmax across full parameter matrix.

Outputs a CSV to $RESULTS_DIR/final_benchmark.csv with columns:
  N, D, causal, q, mode, provider, tflops, ms, gbs, B, H
"""

import os
import csv
import torch
import triton

from stieltjes_flash_attn import stieltjes_attention

DEVICE = torch.device("cuda")

# Fixed dimensions
B = 4
H = 8

# Benchmark matrix
N_VALS    = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
D_VALS    = [64, 128]
CAUSAL    = [False, True]
Q_VALS    = [1.0, 2.0, 4.0, 8.0]   # Stieltjes only
MODES     = ["fwd", "bwd"]
PROVIDERS = ["stieltjes", "softmax"]


def flops(B, H, N, D, mode, causal):
    total = 2 * 2.0 * B * H * N * N * D
    if causal:
        total *= 0.5
    if mode == "bwd":
        total *= 2.5
    return total


def gbs(B, H, N, D, ms):
    """Approximate memory bandwidth: read Q,K,V + write O, each BxHxNxD fp16."""
    return 2 * B * H * N * D * 2 * 1e-9 / (ms * 1e-3)


def run_softmax(B, H, N, D, causal, mode):
    dtype = torch.float16
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=(mode == "bwd"))
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=(mode == "bwd"))
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=(mode == "bwd"))
    sm_scale = D ** -0.5

    def fwd():
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        if causal:
            mask = torch.triu(torch.ones(N, N, device=DEVICE, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    if mode == "fwd":
        fn = fwd
    else:
        o = fwd()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn)
    return ms


def run_stieltjes(B, H, N, D, causal, mode, sq):
    dtype = torch.float16
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=(mode == "bwd"))
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=(mode == "bwd"))
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=(mode == "bwd"))
    sm_scale = D ** -0.5

    def fwd():
        return stieltjes_attention(q, k, v, causal=causal, sm_scale=sm_scale,
                                   stieltjes_q=sq, num_iter=3)

    if mode == "fwd":
        fn = fwd
    else:
        o = fwd()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn)
    return ms


def main():
    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "final_benchmark.csv")

    fieldnames = ["N", "D", "causal", "q", "mode", "provider", "tflops", "ms", "gbs", "B", "H"]

    total_configs = (
        len(N_VALS) * len(D_VALS) * len(CAUSAL) * len(MODES) *
        (len(PROVIDERS) - 1 + len(Q_VALS))  # softmax once, stieltjes per q
    )
    done = 0

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for N in N_VALS:
            for D in D_VALS:
                for causal in CAUSAL:
                    for mode in MODES:
                        # --- softmax ---
                        done += 1
                        tag = f"[{done}/{total_configs}] N={N} D={D} causal={causal} mode={mode} provider=softmax q=-"
                        print(tag, flush=True)
                        try:
                            ms = run_softmax(B, H, N, D, causal, mode)
                            tf = flops(B, H, N, D, mode, causal) * 1e-12 / (ms * 1e-3)
                            gb = gbs(B, H, N, D, ms)
                            writer.writerow(dict(
                                N=N, D=D, causal=causal, q="-", mode=mode,
                                provider="softmax", tflops=f"{tf:.4f}",
                                ms=f"{ms:.4f}", gbs=f"{gb:.4f}", B=B, H=H,
                            ))
                        except torch.cuda.OutOfMemoryError:
                            print(f"  OOM", flush=True)
                            writer.writerow(dict(
                                N=N, D=D, causal=causal, q="-", mode=mode,
                                provider="softmax", tflops="OOM",
                                ms="OOM", gbs="OOM", B=B, H=H,
                            ))
                            torch.cuda.empty_cache()

                        fh.flush()

                        # --- stieltjes ---
                        for sq in Q_VALS:
                            done += 1
                            tag = f"[{done}/{total_configs}] N={N} D={D} causal={causal} mode={mode} provider=stieltjes q={sq}"
                            print(tag, flush=True)
                            try:
                                ms = run_stieltjes(B, H, N, D, causal, mode, sq)
                                tf = flops(B, H, N, D, mode, causal) * 1e-12 / (ms * 1e-3)
                                gb = gbs(B, H, N, D, ms)
                                writer.writerow(dict(
                                    N=N, D=D, causal=causal, q=sq, mode=mode,
                                    provider="stieltjes", tflops=f"{tf:.4f}",
                                    ms=f"{ms:.4f}", gbs=f"{gb:.4f}", B=B, H=H,
                                ))
                            except torch.cuda.OutOfMemoryError:
                                print(f"  OOM", flush=True)
                                writer.writerow(dict(
                                    N=N, D=D, causal=causal, q=sq, mode=mode,
                                    provider="stieltjes", tflops="OOM",
                                    ms="OOM", gbs="OOM", B=B, H=H,
                                ))
                                torch.cuda.empty_cache()

                            fh.flush()

    print(f"\nDone. Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
