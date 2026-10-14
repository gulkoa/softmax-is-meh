"""GPU tests for the ALiBi bias path of the fused Stieltjes kernel (stjalibi
worktree). Run on one H100:  python test_alibi_kernel.py
(a) ALIBI off: this kernel is bit-identical to main's kernel (fwd + bwd).
(b) ALIBI on: fwd/bwd vs an fp64 reference (bisection λ + one Newton step with
    grad = the IFT gradient), B>1 with distinct per-head slopes, N not a
    multiple of the block, NR-8 and Halley-3.
(c) strided fused-qkv views give the same result as contiguous inputs.
(d) λ convergence: max/mean |Σw − 1| at N=1024 and 8192, with and without
    the bias, Halley 3/4/5 (the init (i+1)^{1/q} ignores the bias).
"""
import importlib.util, math, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_K = "/users/PAS2402/alexg/softmax/softmax-is-meh/triton/stieltjes_flash_attn.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


KW = load(os.path.join(HERE, "stieltjes_flash_attn.py"), "k_worktree")
KM = load(MAIN_K, "k_main")
dev = torch.device("cuda")
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def slopes_for(H):
    h = torch.arange(1, H + 1, dtype=torch.float32)
    return 2.0 ** (-8.0 * h / H)


def run(mod, q, k, v, do, **kw):
    q = q.detach().clone().requires_grad_(True); k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    o = mod.stieltjes_attention(q, k, v, causal=True, **kw)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def ref(q, k, v, sm_scale, qord, slopes):
    """fp64: biased causal scores; λ by bisection (no grad) + one Newton step with grad."""
    B, H, N, D = q.shape
    s = (q @ k.transpose(-1, -2)) * sm_scale
    i = torch.arange(N, device=q.device, dtype=torch.float64)
    dist = (i[:, None] - i[None, :])
    if slopes is not None:
        s = s - slopes.to(q.device, torch.float64)[None, :, None, None] * dist.clamp(min=0)
    mask = dist >= 0
    s = s.masked_fill(~mask, float("-inf"))
    with torch.no_grad():
        smax = s.max(-1, keepdim=True).values
        lo, hi = smax + 1e-12, smax + (N ** (1.0 / qord)) + 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f = (mid - s).clamp_min(1e-300).pow(-qord).masked_fill(~mask, 0).sum(-1, keepdim=True)
            gt = f > 1.0
            lo = torch.where(gt, mid, lo); hi = torch.where(gt, hi, mid)
        lam0 = 0.5 * (lo + hi)
    d = (lam0 - s).clamp_min(1e-300)
    f = d.pow(-qord).masked_fill(~mask, 0).sum(-1, keepdim=True) - 1.0
    fp = (-qord) * d.pow(-qord - 1).masked_fill(~mask, 0).sum(-1, keepdim=True)
    lam = lam0 - f / fp
    w = (lam - s).clamp_min(1e-300).pow(-qord).masked_fill(~mask, 0)
    p = w / w.sum(-1, keepdim=True)
    return p @ v


def rel(a, b):
    return float((a.double() - b.double()).abs().max() / (b.double().abs().max() + 1e-12))


torch.manual_seed(0)
# ---------- (a) ALIBI off: bit-identical to main's kernel ----------
for (B, H, N, D, dt) in ((2, 12, 1024, 64, torch.bfloat16), (2, 4, 200, 64, torch.float32)):
    q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=dt) for _ in range(3))
    do = torch.randn(B, H, N, D, device=dev, dtype=dt)
    kw = dict(stieltjes_q=4.0, num_iter=3, solver="halley", normalize=True, ift_grad=True)
    a = run(KW, q, k, v, do, **kw); b = run(KM, q, k, v, do, **kw)
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    check(f"(a) ALIBI off == main kernel, bit-identical, {dt} B{B} H{H} N{N}", same)

# ---------- (b) ALIBI on vs fp64 reference ----------
B, H, N, D, qord = 2, 4, 200, 64, 4.0
sm = 1.0 / math.sqrt(D)
sl = slopes_for(H).to(dev)
q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=torch.float32) for _ in range(3))
do = torch.randn(B, H, N, D, device=dev, dtype=torch.float32)
qd, kd, vd = (t.double().detach().requires_grad_(True) for t in (q, k, v))
ro = ref(qd, kd, vd, sm, qord, sl)
ro.backward(do.double())
for solver, it, tol in (("nr", 8, 2e-4), ("halley", 3, 2e-3)):
    o, gq, gk, gv = run(KW, q, k, v, do, stieltjes_q=qord, num_iter=it, solver=solver, normalize=True,
                        ift_grad=True, alibi_slopes=sl)
    errs = (rel(o, ro), rel(gq, qd.grad), rel(gk, kd.grad), rel(gv, vd.grad))
    check(f"(b) ALIBI on vs fp64 ref, {solver}-{it}: rel err o/dq/dk/dv", max(errs) < tol,
          " ".join(f"{e:.1e}" for e in errs) + f" (tol {tol:.0e})")
# per-head slopes really differ: swap the slopes -> the output must change for heads 0 and 3
o_sw = run(KW, q, k, v, do, stieltjes_q=qord, num_iter=8, solver="nr", normalize=True, ift_grad=True,
           alibi_slopes=sl.flip(0))[0]
o_n = run(KW, q, k, v, do, stieltjes_q=qord, num_iter=8, solver="nr", normalize=True, ift_grad=True,
          alibi_slopes=sl)[0]
check("(b) per-head slopes are indexed by head (flipped slopes change heads 0 and 3)",
      (o_sw[:, 0] - o_n[:, 0]).abs().max() > 1e-3 and (o_sw[:, 3] - o_n[:, 3]).abs().max() > 1e-3)
# batch index does not leak into the slope (B=2: batch 1 must equal a B=1 run on the same data)
o_b1 = run(KW, q[1:].contiguous(), k[1:].contiguous(), v[1:].contiguous(), do[1:].contiguous(),
           stieltjes_q=qord, num_iter=8, solver="nr", normalize=True, ift_grad=True, alibi_slopes=sl)[0]
check("(b) batch 1 of a B=2 call == the same rows run alone (slope uses off_h, not off_hz)",
      torch.allclose(o_n[1:], o_b1, atol=1e-5, rtol=1e-5))

# ---------- (c) strided fused-qkv views ----------
x = torch.randn(B, N, 3 * H * D, device=dev, dtype=torch.float32)
qs, ks, vs = x.split(H * D, dim=2)
qs, ks, vs = (t.view(B, N, H, D).transpose(1, 2) for t in (qs, ks, vs))
assert not qs.is_contiguous()
kw = dict(stieltjes_q=qord, num_iter=3, solver="halley", normalize=True, ift_grad=True, alibi_slopes=sl)
a = run(KW, qs, ks, vs, do, **kw)
b = run(KW, qs.contiguous(), ks.contiguous(), vs.contiguous(), do, **kw)
check("(c) strided fused-qkv views == contiguous (fwd + grads)", all(torch.allclose(x_, y_, atol=1e-5, rtol=1e-5) for x_, y_ in zip(a, b)))

# ---------- (d) λ convergence: |Σw − 1| ----------
class _Ctx:
    def save_for_backward(self, *t):
        self.saved = t


def residual(N, use_alibi, iters, H=12, scale_mult=2.0):
    torch.manual_seed(1)
    q = torch.randn(1, H, N, 64, device=dev, dtype=torch.bfloat16) * scale_mult
    k = torch.randn(1, H, N, 64, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, H, N, 64, device=dev, dtype=torch.bfloat16)
    c = _Ctx()
    KW.StieltjesAttention.forward(c, q, k, v, True, 1 / 8, 4.0, iters, False, True, True, "halley",
                                  slopes_for(H).to(dev) if use_alibi else None)
    wsum = c.saved[7]
    r = (wsum - 1.0).abs()
    return float(r.max()), float(r.mean())


print("(d) λ residual |Σw − 1| (bf16, H=12, q=4, scores ~ N(0, 4))", flush=True)
for N in (1024, 8192):
    for it in (3, 4, 5):
        off = residual(N, False, it); on = residual(N, True, it)
        print(f"    N={N:5d} halley-{it}: no bias max {off[0]:.1e} mean {off[1]:.1e} | ALiBi max {on[0]:.1e} mean {on[1]:.1e}", flush=True)

print("FAILED:", fails if fails else "none", flush=True)
sys.exit(1 if fails else 0)
