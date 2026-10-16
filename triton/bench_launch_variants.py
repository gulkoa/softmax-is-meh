"""Launch-setting sweep for every Stieltjes kernel, through the public API.

For each kernel (fwd, delta, dkdv, dq) and each (num_warps, num_stages) in
{4, 8} x {1, 2, 3}, sets stieltjes_flash_attn.LAUNCH[kernel] with the other
kernels at Triton's defaults, runs forward + backward under torch.profiler
and reports that kernel's time per call. Each variant's output and dq, dk,
dv are compared bit for bit with the default launch. Then all four kernels
are set to their fastest variant at the first shape and the whole
forward + backward is timed at every shape. Production recipe: bf16, causal,
q 4, Halley-3, normalized, IFT gradient, D 64.

    python triton/bench_launch_variants.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stieltjes_flash_attn as K  # noqa: E402

assert K.__file__.startswith(HERE), K.__file__
NAMES = {"fwd": "_stieltjes_attn_fwd", "delta": "_stieltjes_bwd_delta",
         "dkdv": "_stieltjes_bwd_dkdv", "dq": "_stieltjes_bwd_dq"}
VARIANTS = [(nw, ns) for nw in (4, 8) for ns in (1, 2, 3)]
SHAPES = [(8, 16, 1024, 64), (16, 12, 1024, 64), (2, 16, 4096, 64), (1, 16, 8192, 64)]


def bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def inputs(B, H, N, D):
    g = torch.Generator(device="cuda").manual_seed(N + B)
    return [torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16, generator=g)
            for _ in range(4)]


def run_once(q, k, v, do):
    q_, k_, v_ = (t.detach().clone().requires_grad_() for t in (q, k, v))
    o = K.stieltjes_attention(q_, k_, v_, causal=True, sm_scale=1.0 / math.sqrt(q.shape[-1]),
                              stieltjes_q=4.0, num_iter=3, solver="halley",
                              normalize=True, ift_grad=True)
    o.backward(do)
    return o.detach(), q_.grad, k_.grad, v_.grad


def kernel_times(x, iters):
    for _ in range(2):
        run_once(*x)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run_once(*x)
        torch.cuda.synchronize()
    t = dict.fromkeys(NAMES, 0.0)
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        for n, kn in NAMES.items():
            if kn in ev.name:
                t[n] += ev.time_range.elapsed_us()
    return {n: us / 1000 / iters for n, us in t.items()}


def same(a, b):
    return all(torch.equal(bits(x), bits(y)) for x, y in zip(a, b))


def main():
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}", flush=True)
    best = {}
    for si, shape in enumerate(SHAPES):
        iters = 10 if shape[2] <= 1024 else 4
        for n in NAMES:
            K.LAUNCH[n] = {}
        x = inputs(*shape)
        ref = run_once(*x)
        base = kernel_times(x, iters)
        print(f"\nshape {shape}: default launch " + ", ".join(
            f"{n} {t:.3f}" for n, t in base.items()) + f" ms (sum {sum(base.values()):.3f})",
            flush=True)
        for n in NAMES:
            rows = []
            for nw, ns in VARIANTS:
                K.LAUNCH[n] = dict(num_warps=nw, num_stages=ns)
                ident = same(ref, run_once(*x))
                t = kernel_times(x, iters)[n]
                rows.append((t, nw, ns, ident))
            K.LAUNCH[n] = {}
            print(f"  {n:5s} " + "  ".join(f"w{nw}s{ns} {t:.3f}{'' if ident else ' (DIFF)'}"
                                          for t, nw, ns, ident in rows), flush=True)
            if si == 0:
                t, nw, ns, ident = min(r for r in rows if r[3])
                best[n] = dict(num_warps=nw, num_stages=ns)
        if si == 0:
            print("  fastest bit-identical at this shape:", best, flush=True)
        for n in NAMES:
            K.LAUNCH[n] = dict(best[n])
        tuned = kernel_times(x, iters)
        ident = same(ref, run_once(*x))
        print(f"  with those settings: " + ", ".join(f"{n} {t:.3f}" for n, t in tuned.items())
              + f" ms (sum {sum(tuned.values()):.3f}, {sum(base.values()) / sum(tuned.values()):.2f}x"
              f" the default; bit-identical {ident})", flush=True)


if __name__ == "__main__":
    main()
