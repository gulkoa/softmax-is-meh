"""Benchmark Stieltjes attention: Triton kernel vs PyTorch reference.

This is the workshop-paper headline: how much speedup does our Triton kernel
deliver over the naive PyTorch implementation of the same algorithm?

Output CSV: $RESULTS_DIR/bench_triton_vs_ref_ascend_a100.csv
Columns: B, H, N, D, q, causal, mode,
         fwd_ms_triton, fwd_ms_ref, bwd_ms_triton, bwd_ms_ref,
         speedup_fwd, speedup_bwd

Mode column is redundant with the split columns but included for easy pivoting.

Run on Ascend A100 only. Do not mix hardware results with Cardinal H100.
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

from stieltjes_flash_attn import (  # noqa: E402
    stieltjes_attention,
    stieltjes_attention_ref,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16  # bf16 has wider exponent range — masked-fill sentinel value
                          # (finfo.min) is representable; fp16 overflows on causal mask.

# Benchmark matrix — chosen to cover typical training shapes without OOM on A100-40GB.
SHAPES = [
    # (B, H, N, D)
    (1, 8, 128, 64),
    (1, 8, 512, 64),
    (1, 8, 1024, 64),
    (1, 8, 2048, 64),
    (1, 8, 4096, 64),
    (4, 8, 128, 64),
    (4, 8, 512, 64),
    (4, 8, 1024, 64),
    (4, 8, 2048, 64),
    (1, 8, 1024, 128),
    (1, 8, 2048, 128),
]
Q_VALS = [2.0, 4.0]       # representative "trainable" sharpness
CAUSAL = [False, True]
WARMUP = 25
REPEAT = 100


def bench_fwd(fn) -> float:
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REPEAT)


def bench_bwd(fn_and_out) -> float:
    """fn_and_out() must return (output_tensor,) that still has grad_fn."""
    # Build once; do_bench will re-invoke backward many times. Must keep graph.
    def run():
        out = fn_and_out()
        # Fresh grad each iter; retain_graph=True to reuse the same graph.
        out.backward(torch.ones_like(out), retain_graph=True)
    return triton.testing.do_bench(run, warmup=WARMUP, rep=REPEAT)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None,
                   help="Output CSV path. Default: $RESULTS_DIR/bench_triton_vs_ref_ascend_a100.csv. "
                        "Pass an explicit filename when running on non-A100 hardware to avoid clobbering.")
    args = p.parse_args()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "bench_triton_vs_ref_ascend_a100.csv"

    rows = []
    for (B, H, N, D) in SHAPES:
        for sq in Q_VALS:
            for causal in CAUSAL:
                sm_scale = 1.0 / (D ** 0.5)
                torch.manual_seed(0)

                q_t = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
                k_t = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
                v_t = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

                # Separate tensors for ref so grads don't accumulate across impls.
                q_r = q_t.detach().clone().requires_grad_()
                k_r = k_t.detach().clone().requires_grad_()
                v_r = v_t.detach().clone().requires_grad_()

                # --- Forward timing (no grad for cleanliness) ---
                def fwd_triton():
                    return stieltjes_attention(q_t, k_t, v_t, causal=causal,
                                               sm_scale=sm_scale, stieltjes_q=sq)

                def fwd_ref():
                    return stieltjes_attention_ref(q_r, k_r, v_r, sm_scale=sm_scale,
                                                   causal=causal, stieltjes_q=sq)

                try:
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        ms_fwd_t = bench_fwd(fwd_triton)
                    peak_mb_t = torch.cuda.max_memory_allocated() / (1024 * 1024)
                except Exception as e:
                    print(f"SKIP fwd triton B={B} H={H} N={N} D={D} q={sq} c={causal}: {e}")
                    ms_fwd_t = float("nan")
                    peak_mb_t = float("nan")
                try:
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        ms_fwd_r = bench_fwd(fwd_ref)
                    peak_mb_r = torch.cuda.max_memory_allocated() / (1024 * 1024)
                except Exception as e:
                    print(f"SKIP fwd ref B={B} H={H} N={N} D={D} q={sq} c={causal}: {e}")
                    ms_fwd_r = float("nan")
                    peak_mb_r = float("nan")

                # --- Backward timing ---
                # CLAUDE.md note: triton kernel backward has NaN — don't bench it.
                # We keep "fwd speedup" as the headline number.
                ms_bwd_t = float("nan")
                try:
                    ms_bwd_r = bench_bwd(fwd_ref)
                except Exception as e:
                    print(f"SKIP bwd ref B={B} H={H} N={N} D={D} q={sq} c={causal}: {e}")
                    ms_bwd_r = float("nan")

                speedup_fwd = (ms_fwd_r / ms_fwd_t) if (ms_fwd_t and ms_fwd_t == ms_fwd_t) else float("nan")
                speedup_bwd = float("nan")

                row = {
                    "B": B, "H": H, "N": N, "D": D, "q": sq, "causal": causal,
                    "fwd_ms_triton": f"{ms_fwd_t:.4f}",
                    "fwd_ms_ref":    f"{ms_fwd_r:.4f}",
                    "bwd_ms_triton": f"{ms_bwd_t:.4f}",
                    "bwd_ms_ref":    f"{ms_bwd_r:.4f}",
                    "speedup_fwd":   f"{speedup_fwd:.3f}",
                    "speedup_bwd":   f"{speedup_bwd:.3f}",
                    "peak_mb_triton": f"{peak_mb_t:.1f}",
                    "peak_mb_ref":    f"{peak_mb_r:.1f}",
                }
                rows.append(row)
                print(f"  B={B} H={H} N={N:5d} D={D} q={sq} c={int(causal)}  "
                      f"fwd: triton={ms_fwd_t:.3f}ms ref={ms_fwd_r:.3f}ms  "
                      f"speedup={speedup_fwd:.2f}x")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
