"""
Head-to-head performance + memory benchmark: Triton FLASHn (normalize=True) vs
Jack's actual normalized StieltjesTransform (bisection, PyTorch,
architecture_new branch).

Protocol (matches the April longctx suite): B=1, H=8, D=64, q=4, non-causal
(Jack's mapping has no causal path). Providers:
  jack-stj      : scores = qk^T*scale -> Jack's translate_logits -> @v, fp32
  flashn-fp32   : stieltjes_attention(..., normalize=True), fp32
  flashn-fp16   : same, fp16 inputs

Captured per (provider, N):
  - fwd latency ms (no_grad) and fwd+bwd latency ms (full training step shape)
  - peak CUDA memory (allocated AND reserved) for fwd and fwd+bwd
  - graph-held memory: bytes alive after a requires_grad forward, before
    backward — i.e. the autograd saved-tensor footprint (the O(N^2) part for
    Jack; O(N) for the kernel)
  - input-tensor baseline memory (so overhead over inputs is computable)
  - OOM points (caught per provider per N)

Logging: CSV to stdout, JSON to results/, and wandb (offline mode on compute
nodes — sync with `wandb sync` from the login node). One wandb run per
benchmark invocation; per-(provider,N) metrics logged as steps plus a final
summary table.
"""
import json
import os
import sys

import torch

WORKTREE = "/users/PAS2402/alexg/softmax/psm-architecture-new"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")

from mappings.stieltjes import StieltjesTransform as JackNorm  # noqa: E402
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402

B, H, D, SQ = 1, 8, 64, 4.0
# NR iters: 8 is fully converged for q<=16 (job 12312766 sensitivity study);
# 20 was the original conservative config. Env-overridable for A/B runs.
NUM_ITER_TRITON = int(os.environ.get("BENCH_NUM_ITER", "8"))
# Env-overridable sweep (comma-separated) so tail runs can skip slow cells
# (fp32-IEEE at N>=16k takes ~25s/iter and blows the walltime).
NS = [int(x) for x in os.environ.get(
    "BENCH_NS", "1024,2048,4096,8192,16384,32768,65536,131072").split(",")]
PROVIDERS = os.environ.get(
    "BENCH_PROVIDERS", "jack-stj,flashn-fp32,flashn-fp16").split(",")
OUT_JSON = os.environ.get(
    "BENCH_OUT",
    "/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_flashn_vs_jack.json")


def _time_fn(fn, warmup=3, iters=10):
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
    return start.elapsed_time(end) / iters


def make_inputs(provider, N, requires_grad):
    device = torch.device("cuda")
    dtype = torch.float16 if provider == "flashn-fp16" else torch.float32
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def make_fwd(provider, q, k, v):
    sm = 1.0 / D ** 0.5
    if provider == "jack-stj":
        mapping = JackNorm(q=SQ)  # his defaults: num_iter=16, eps=1e-9

        def fwd():
            scores = torch.matmul(q, k.transpose(-2, -1)) * sm
            p = mapping.translate_logits(scores, dim=-1)
            return torch.matmul(p, v)
    else:
        def fwd():
            return stieltjes_attention(q, k, v, causal=False, sm_scale=sm,
                                       stieltjes_q=SQ, num_iter=NUM_ITER_TRITON,
                                       normalize=True)
    return fwd


def bench_cell(provider, N):
    """Full capture for one (provider, N). Returns dict; marks OOM per phase."""
    r = {"provider": provider, "N": N}
    torch.cuda.empty_cache()

    # --- baseline: input tensors only ---
    try:
        torch.cuda.reset_peak_memory_stats()
        q, k, v = make_inputs(provider, N, requires_grad=False)
        torch.cuda.synchronize()
        r["input_mem_mb"] = torch.cuda.memory_allocated() / 1e6
    except torch.cuda.OutOfMemoryError:
        r["input_mem_mb"] = "OOM"
        return r

    # --- forward only (no_grad) ---
    fwd = make_fwd(provider, q, k, v)
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            r["fwd_ms"] = _time_fn(fwd)
        r["fwd_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 1e6
        r["fwd_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
    except torch.cuda.OutOfMemoryError:
        r["fwd_ms"] = r["fwd_peak_alloc_mb"] = r["fwd_peak_reserved_mb"] = "OOM"
        del q, k, v
        return r
    del q, k, v
    torch.cuda.empty_cache()

    # --- graph-held memory: forward with autograd, measure before backward ---
    try:
        q, k, v = make_inputs(provider, N, requires_grad=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        fwd = make_fwd(provider, q, k, v)
        out = fwd()
        torch.cuda.synchronize()
        r["graph_held_mb"] = (torch.cuda.memory_allocated() - base) / 1e6
        del out
    except torch.cuda.OutOfMemoryError:
        r["graph_held_mb"] = "OOM"
        try:
            del q, k, v
        except NameError:
            pass
        torch.cuda.empty_cache()
        # continue: fwd+bwd will also OOM but let it record so
    # --- fwd+bwd (training-step shape) ---
    try:
        torch.cuda.empty_cache()
        q, k, v = make_inputs(provider, N, requires_grad=True)
        fwd = make_fwd(provider, q, k, v)
        dtype = q.dtype
        do = torch.randn(B, H, N, D, device=q.device, dtype=dtype)

        def step():
            q.grad = k.grad = v.grad = None
            fwd().backward(do)

        torch.cuda.reset_peak_memory_stats()
        r["fwdbwd_ms"] = _time_fn(step)
        r["fwdbwd_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 1e6
        r["fwdbwd_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
    except torch.cuda.OutOfMemoryError:
        r["fwdbwd_ms"] = r["fwdbwd_peak_alloc_mb"] = r["fwdbwd_peak_reserved_mb"] = "OOM"
    finally:
        torch.cuda.empty_cache()
    return r


def fmt(x, unit=""):
    if isinstance(x, str):
        return x
    return f"{x:.2f}{unit}" if isinstance(x, float) else str(x)


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"B={B} H={H} D={D} q={SQ} non-causal; jack=bisection-16 fp32, "
          f"flashn=NR-{NUM_ITER_TRITON} normalize=True\n")

    # wandb offline (compute nodes have no internet; sync later)
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb
    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"bench-flashn-vs-jack-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config=dict(B=B, H=H, D=D, q=SQ, num_iter_triton=NUM_ITER_TRITON,
                    jack="bisection num_iter=16 fp32 (architecture_new)",
                    gpu=gpu, Ns=NS, providers=PROVIDERS),
    )

    rows = []
    cols = ["provider", "N", "input_mem_mb", "fwd_ms", "fwd_peak_alloc_mb",
            "fwd_peak_reserved_mb", "graph_held_mb", "fwdbwd_ms",
            "fwdbwd_peak_alloc_mb", "fwdbwd_peak_reserved_mb"]
    print(",".join(cols))
    table = wandb.Table(columns=cols)
    for N in NS:
        for prov in PROVIDERS:
            r = bench_cell(prov, N)
            rows.append(r)
            vals = [r.get(c, "") for c in cols]
            print(",".join(fmt(x) for x in vals), flush=True)
            table.add_data(*[str(v) for v in vals])
            # numeric metrics per provider, N as step
            metrics = {f"{prov}/{k}": v for k, v in r.items()
                       if isinstance(v, (int, float)) and k != "N"}
            metrics["N"] = N
            run.log(metrics)

    run.log({"benchmark_table": table})

    with open(OUT_JSON, "w") as f:
        json.dump({"gpu": gpu, "config": dict(B=B, H=H, D=D, q=SQ),
                   "rows": rows}, f, indent=2)
    print(f"\nSaved {OUT_JSON}")

    # Summary tables
    d = {(r["provider"], r["N"]): r for r in rows}

    def cell(prov, N, key):
        return fmt(d.get((prov, N), {}).get(key, "?"))

    print("\n===== fwd+bwd latency ms (speedup = jack / flashn) =====")
    print(f"{'N':>7} | {'jack':>10} | {'flashn32':>10} | {'flashn16':>10} | "
          f"{'sp32':>6} | {'sp16':>6}")
    for N in NS:
        j = d.get(("jack-stj", N), {}).get("fwdbwd_ms", "?")
        f32 = d.get(("flashn-fp32", N), {}).get("fwdbwd_ms", "?")
        f16 = d.get(("flashn-fp16", N), {}).get("fwdbwd_ms", "?")

        def sp(a, b):
            return (f"{a/b:.1f}x" if isinstance(a, float) and isinstance(b, float)
                    else "-")
        print(f"{N:>7} | {fmt(j):>10} | {fmt(f32):>10} | {fmt(f16):>10} | "
              f"{sp(j, f32):>6} | {sp(j, f16):>6}")

    print("\n===== fwd+bwd peak allocated MB =====")
    print(f"{'N':>7} | {'jack':>10} | {'flashn32':>10} | {'flashn16':>10}")
    for N in NS:
        print(f"{N:>7} | {cell('jack-stj', N, 'fwdbwd_peak_alloc_mb'):>10} | "
              f"{cell('flashn-fp32', N, 'fwdbwd_peak_alloc_mb'):>10} | "
              f"{cell('flashn-fp16', N, 'fwdbwd_peak_alloc_mb'):>10}")

    print("\n===== graph-held (autograd saved-tensor) MB =====")
    print(f"{'N':>7} | {'jack':>10} | {'flashn32':>10} | {'flashn16':>10}")
    for N in NS:
        print(f"{N:>7} | {cell('jack-stj', N, 'graph_held_mb'):>10} | "
              f"{cell('flashn-fp32', N, 'graph_held_mb'):>10} | "
              f"{cell('flashn-fp16', N, 'graph_held_mb'):>10}")

    run.finish()
    print("\nwandb offline run saved; sync from login node with: "
          "wandb sync <run dir printed above>")


if __name__ == "__main__":
    main()
