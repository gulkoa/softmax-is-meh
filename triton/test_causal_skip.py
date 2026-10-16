"""Causal tile skipping in the forward and delta kernels must not change a bit.

Loads the pre-change kernel from git (commit dc8dd3d, main before the
change) next to this checkout's kernel and compares, for causal and
non-causal inputs, shapes with N a multiple of the tile and not, every
supported dtype path and the solver / normalization / gradient modes:
the output and dq, dk, dv with torch.equal. Then times both at the 355M
and 124M training shapes and at N = 4,096 / 8,192 (forward, forward + backward).

    python triton/test_causal_skip.py [--bench]
"""
import argparse
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF_COMMIT = "dc8dd3d"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_module(tmp):
    src = subprocess.run(["git", "-C", HERE, "show",
                          f"{REF_COMMIT}:triton/stieltjes_flash_attn.py"],
                         capture_output=True, text=True, check=True).stdout
    assert "tl.range(0, hi, BLOCK_N)" not in src.split("def _stieltjes_bwd_dkdv(")[0], \
        "reference already has the change"
    path = os.path.join(tmp, "stj_ref_dc8dd3d.py")
    with open(path, "w") as f:
        f.write(src)
    return load(path, "stj_ref_dc8dd3d")


CASES = [  # (B, H, N, D, dtype)
    (2, 3, 1024, 64, torch.bfloat16),
    (1, 2, 1000, 64, torch.bfloat16),    # N not a multiple of BLOCK_M = 128
    (2, 2, 256, 128, torch.bfloat16),    # 64 x 64 tiles
    (1, 1, 2048, 64, torch.float16),
    (1, 2, 200, 128, torch.float32),     # 32 x 32 tiles
    (3, 2, 64, 32, torch.bfloat16),      # one tile
    (1, 1, 130, 16, torch.float32),
]
MODES = [  # kwargs for stieltjes_attention
    dict(stieltjes_q=4.0, num_iter=3, solver="halley", normalize=True, ift_grad=True),
    dict(stieltjes_q=2.0, num_iter=3, solver="halley", normalize=True, ift_grad=True),
    dict(stieltjes_q=8.0, num_iter=8, solver="nr", normalize=False, ift_grad=False),
    dict(stieltjes_q=1.0, num_iter=8, solver="nr", block_lambda_grad=True),
]


def bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def run(mod, q, k, v, do, causal, kw):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    o = mod.stieltjes_attention(q, k, v, causal=causal,
                                sm_scale=1.0 / math.sqrt(q.shape[-1]), **kw)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def identity(ref, new):
    n_fail = 0
    for B, H, N, D, dt in CASES:
        g = torch.Generator(device="cuda").manual_seed(N * 7 + D)
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=dt, generator=g) * s
                   for s in (2.0, 2.0, 1.0))
        do = torch.randn(B, H, N, D, device="cuda", dtype=dt, generator=g)
        for causal in (True, False):
            for kw in MODES:
                a = run(ref, q, k, v, do, causal, kw)
                b = run(new, q, k, v, do, causal, kw)
                # bitwise (NaN payloads included); a non-finite value only
                # fails when the reference had a finite one there
                same = [torch.equal(bits(x), bits(y)) for x, y in zip(a, b)]
                new_bad = any((~torch.isfinite(y) & torch.isfinite(x)).any().item()
                              for x, y in zip(a, b))
                ok = all(same) and not new_bad
                n_fail += not ok
                if not all(torch.isfinite(x).all().item() for x in a):
                    print(f"note {B},{H},{N},{D} {dt} causal={causal} {kw}: the "
                          f"reference itself is non-finite here (identical bits: {all(same)})")
                if not ok:
                    diffs = [(x.float() - y.float()).abs().max().item() for x, y in zip(a, b)]
                    print(f"FAIL {B},{H},{N},{D} {dt} causal={causal} {kw}: bit-equal "
                          f"o/dq/dk/dv {same}, new non-finite {new_bad}, max|diff| {diffs}")
    total = len(CASES) * 2 * len(MODES)
    print(f"bit-identity: {total - n_fail}/{total} cases identical")
    return n_fail == 0


def time_ms(fn, iters=30):
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


def bench(ref, new):
    kw = MODES[0]
    for B, H, N in ((8, 16, 1024), (16, 12, 1024), (2, 16, 4096), (1, 16, 8192)):
        D = 64
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v, do = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16,
                                   generator=g) for _ in range(4))
        q, k, v = (t.requires_grad_() for t in (q, k, v))
        sm = 1.0 / math.sqrt(D)
        res = {}
        for name, mod in (("ref", ref), ("new", new)):
            def fwd():
                with torch.no_grad():
                    mod.stieltjes_attention(q, k, v, causal=True, sm_scale=sm, **kw)

            def fb():
                mod.stieltjes_attention(q, k, v, causal=True, sm_scale=sm, **kw).backward(do)
                q.grad = k.grad = v.grad = None
            res[name] = (time_ms(fwd, 60), time_ms(fb, 30))
        (rf, rfb), (nf, nfb) = res["ref"], res["new"]
        print(f"shape {B},{H},{N},{D}: fwd {rf:.3f} -> {nf:.3f} ms ({rf / nf:.2f}x); "
              f"bwd {rfb - rf:.3f} -> {nfb - nf:.3f} ms ({(rfb - rf) / (nfb - nf):.2f}x); "
              f"fwd+bwd {rfb:.3f} -> {nfb:.3f} ms ({rfb / nfb:.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP - no CUDA available")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ref = ref_module(tmp)
        new = load(os.path.join(HERE, "stieltjes_flash_attn.py"), "stj_new")
        ok = identity(ref, new)
        if a.bench:
            bench(ref, new)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
