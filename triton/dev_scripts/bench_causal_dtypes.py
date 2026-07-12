"""
SPEED: repeat the April longctx-scaling benchmark with the NEW kernel.

Providers:
  flashn-fp16 / flashn-bf16 : normalized Triton kernel (normalize=True)
  flash-ift-fp16            : default (IFT) mode, for regression vs April numbers
  ref-stj-fp32              : PyTorch NR reference (stieltjes_attention_ref),
                              materializes N^2 (the April baseline)
  sdpa-fp16                 : torch flash softmax (speed-of-light reference)

Sweep: causal in {False, True}, N in {1024, 4096, 16384, 65536}.
Metrics: fwd latency, fwd+bwd latency, peak allocated/reserved memory, OOM.
wandb-logged incrementally.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")

from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref  # noqa: E402

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

B, H, D, SQ = 1, 8, 64, 4.0
NUM_ITER = int(os.environ.get("BENCH_NUM_ITER", "8"))
NS = [int(x) for x in os.environ.get(
    "BENCH_NS", "1024,4096,16384,65536").split(",")]
PROVIDERS = os.environ.get(
    "BENCH_PROVIDERS",
    "flashn-fp16,flashn-bf16,flash-ift-fp16,ref-stj-fp32,sdpa-fp16").split(",")
DTYPES = {"flashn-fp16": torch.float16, "flashn-bf16": torch.bfloat16,
          "flash-ift-fp16": torch.float16, "ref-stj-fp32": torch.float32,
          "sdpa-fp16": torch.float16}


def _time_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def make_fwd(provider, q, k, v, causal):
    sm = 1.0 / D ** 0.5
    if provider.startswith("flashn"):
        return lambda: stieltjes_attention(q, k, v, causal=causal, sm_scale=sm,
                                           stieltjes_q=SQ, num_iter=NUM_ITER,
                                           normalize=True)
    if provider == "flash-ift-fp16":
        return lambda: stieltjes_attention(q, k, v, causal=causal, sm_scale=sm,
                                           stieltjes_q=SQ, num_iter=NUM_ITER)
    if provider == "ref-stj-fp32":
        return lambda: stieltjes_attention_ref(q, k, v, sm, causal=causal,
                                               stieltjes_q=SQ, num_iter=NUM_ITER)
    if provider == "sdpa-fp16":
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    raise ValueError(provider)


def bench_cell(provider, N, causal):
    r = {"provider": provider, "N": N, "causal": causal}
    device = torch.device("cuda")
    dtype = DTYPES[provider]
    torch.cuda.empty_cache()
    for mode in ["fwd", "fwdbwd"]:
        try:
            torch.cuda.reset_peak_memory_stats()
            rg = (mode == "fwdbwd")
            q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=rg)
            k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=rg)
            v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=rg)
            fwd = make_fwd(provider, q, k, v, causal)
            if mode == "fwd":
                with torch.no_grad():
                    r["fwd_ms"] = _time_fn(fwd)
            else:
                do = torch.randn(B, H, N, D, device=device, dtype=dtype)

                def step():
                    q.grad = k.grad = v.grad = None
                    fwd().backward(do)
                r["fwdbwd_ms"] = _time_fn(step)
            r[f"{mode}_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 1e6
            r[f"{mode}_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
            del q, k, v
        except torch.cuda.OutOfMemoryError:
            r[f"{mode}_ms" if mode == "fwd" else "fwdbwd_ms"] = "OOM"
            r[f"{mode}_peak_alloc_mb"] = "OOM"
        finally:
            torch.cuda.empty_cache()
    return r


def fmt(x):
    return x if isinstance(x, str) else (f"{x:.2f}" if isinstance(x, float) else str(x))


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"B={B} H={H} D={D} q={SQ}; flash num_iter={NUM_ITER}; ref num_iter={NUM_ITER} fp32\n")
    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"bench-causal-dtypes-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config=dict(B=B, H=H, D=D, q=SQ, Ns=NS, providers=PROVIDERS, gpu=gpu),
    )
    cols = ["provider", "N", "causal", "fwd_ms", "fwd_peak_alloc_mb",
            "fwdbwd_ms", "fwdbwd_peak_alloc_mb"]
    print(",".join(cols))
    rows = []
    for causal in [False, True]:
        for N in NS:
            for prov in PROVIDERS:
                r = bench_cell(prov, N, causal)
                rows.append(r)
                print(",".join(fmt(r.get(c, "")) for c in cols), flush=True)
                run.log({"N": N, "causal": int(causal), "provider": prov,
                         **{f"{prov}/{'causal' if causal else 'noncausal'}/{k}": v
                            for k, v in r.items()
                            if isinstance(v, (int, float)) and k not in
                            ("N", "causal")}})

    d = {(r["provider"], r["N"], r["causal"]): r for r in rows}
    for causal in [False, True]:
        print(f"\n===== fwd+bwd latency ms ({'causal' if causal else 'non-causal'}) =====")
        print(f"{'N':>7} | " + " | ".join(f"{p:>15}" for p in PROVIDERS))
        for N in NS:
            print(f"{N:>7} | " + " | ".join(
                f"{fmt(d[(p, N, causal)].get('fwdbwd_ms', '?')):>15}"
                for p in PROVIDERS))
        print(f"\n===== fwd+bwd peak alloc MB ({'causal' if causal else 'non-causal'}) =====")
        print(f"{'N':>7} | " + " | ".join(f"{p:>15}" for p in PROVIDERS))
        for N in NS:
            print(f"{N:>7} | " + " | ".join(
                f"{fmt(d[(p, N, causal)].get('fwdbwd_peak_alloc_mb', '?')):>15}"
                for p in PROVIDERS))
    run.finish()
    print("\nDONE")


if __name__ == "__main__":
    main()
