"""Forward-kernel code generation and timing, old (dc8dd3d) vs this branch.

Launches _stieltjes_attn_fwd directly at a training shape (default the
355M one: B 8, H 16, N 1,024, D 64; bf16, causal, q 4, Halley-3,
normalized) for tile sizes x num_warps x num_stages. For each variant it
prints registers, spills, shared memory, whether the loads were pipelined
(async copies in the TTGIR), whether the dots became wgmma, the time per
call and the output's max |difference| from the old kernel's default
launch.

    python triton/bench_fwd_variants.py [--B 8 --H 16]
"""
import argparse
import importlib.util
import math
import os
import subprocess
import tempfile

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def launch(mod, q, k, v, bm, bn, nw, ns, num_iter=3):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    lam = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    d_sum, wsum = torch.empty_like(lam), torch.empty_like(lam)
    am = torch.empty((B * H, N), device=q.device, dtype=torch.int32)
    init = torch.arange(1, N + 1, device=q.device, dtype=torch.float32).pow(0.25)
    kw = {}
    if nw is not None:
        kw["num_warps"] = nw
    if ns is not None:
        kw["num_stages"] = ns

    def go():
        return mod._stieltjes_attn_fwd[(triton.cdiv(N, bm), B * H)](
            q, k, v, o, lam, d_sum, am, wsum, init,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            1.0 / math.sqrt(D), N, H,
            sq=4.0, NUM_ITER=num_iter, HALLEY=True, EPS=1e-6, HEAD_DIM=D,
            BLOCK_M=bm, BLOCK_N=bn, CAUSAL=True, NORMALIZE=True, **kw)
    return go, o


def stats(ck):
    out = {}
    for a in ("n_regs", "n_spills"):
        out[a] = getattr(ck, a, None)
    md = getattr(ck, "metadata", None)
    out["shared"] = getattr(md, "shared", None) if md is not None else None
    ttgir = getattr(ck, "asm", {}).get("ttgir", "")
    out["async_copies"] = ttgir.count("async_copy_global_to_local")
    out["wgmma"] = ttgir.count("warp_group_dot")
    out["dots"] = ttgir.count("tt.dot") + ttgir.count("warp_group_dot")
    return out


def time_ms(fn, iters=40):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--N", type=int, default=1024)
    a = ap.parse_args()
    src = subprocess.run(["git", "-C", HERE, "show", "dc8dd3d:triton/stieltjes_flash_attn.py"],
                         capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stj_ref_dc8dd3d.py")
        open(path, "w").write(src)
        ref = load(path, "stj_ref_dc8dd3d")
        new = load(os.path.join(HERE, "stieltjes_flash_attn.py"), "stj_new")
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = (torch.randn(a.B, a.H, a.N, 64, device="cuda", dtype=torch.bfloat16,
                               generator=g) for _ in range(3))
        base_go, base_o = launch(ref, q, k, v, 128, 64, None, None)
        base_go()
        torch.cuda.synchronize()
        base = base_o.clone()
        variants = ([(n, m, 128, 64, nw, ns) for n, m in (("old", ref), ("new", new))
                     for nw in (4, 8) for ns in (None, 1, 2, 3, 4)]
                    + [("new", new, bm, bn, nw, ns) for bm, bn in ((64, 64), (64, 32), (128, 32))
                       for nw in (4, 8) for ns in (None, 2)])
        print(f"shape B {a.B}, H {a.H}, N {a.N}, D 64; bf16 causal q4 Halley-3 normalized")
        print(f"{'kernel':6s} {'tile':>8s} {'warps':>5s} {'stages':>6s} {'regs':>5s} {'spills':>6s} "
              f"{'smem':>7s} {'async':>5s} {'wgmma':>5s} {'ms':>7s} {'max|do|':>9s}")
        for name, mod, bm, bn, nw, ns in variants:
            try:
                go, o = launch(mod, q, k, v, bm, bn, nw, ns)
                ck = go()
                torch.cuda.synchronize()
                ms = time_ms(go)
                st = stats(ck)
                diff = (o.float() - base.float()).abs().max().item()
                print(f"{name:6s} {bm:>4d}x{bn:<3d} {str(nw):>5s} {str(ns):>6s} {str(st['n_regs']):>5s} "
                      f"{str(st['n_spills']):>6s} {str(st['shared']):>7s} {st['async_copies']:>5d} "
                      f"{st['wgmma']:>5d} {ms:7.3f} {diff:9.2e}", flush=True)
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"{name:6s} {bm:>4d}x{bn:<3d} {str(nw):>5s} {str(ns):>6s} ERROR "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
