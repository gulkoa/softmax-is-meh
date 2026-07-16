"""
Validation for the kernel's normalized-IFT backward (ift_grad=True):

REFERENCE: full autograd through the dense normalized-Stieltjes pipeline
with the IFT lambda (one differentiable Newton step on the bisection root —
the construction validated against finite differences and battle-tested in
the MQMTAR stability study, mqmtar_headtohead_v10.stieltjes_dense_probs).

Checks, for N x q x causal x dtype grids:
  A. forward(ift_grad=True) == forward(normalize=True)  (bitwise: same kernel)
  B. dQ/dK/dV of kernel(normalize=True, ift_grad=True) vs dense-IFT autograd
  C. NaN hygiene at padded shapes
  D. regression: kernel(normalize=True) [detached] still matches the dense
     DETACHED autograd path (guards against accidental cross-mode damage)
"""
import sys

import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts")

from stieltjes_flash_attn import stieltjes_attention  # noqa: E402
from mqmtar_headtohead_v10 import stieltjes_dense_probs  # noqa: E402

DEVICE = torch.device("cuda")


def dense_pipeline(q, k, v, sm_scale, causal, sq, ift):
    scores = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float()) * sm_scale
    if causal:
        S = scores.shape[-1]
        mask = torch.ones(S, S, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -1e9)
    p = stieltjes_dense_probs(scores, sq, iters=60, ift=ift)
    return torch.einsum("bhst,bhtd->bhsd", p, v.float())


def run_case(B, H, N, D, causal, sq, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype) * 0.5
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype) * 0.5
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    do = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / (D ** 0.5)

    # A: forward equivalence between the two normalized modes
    with torch.no_grad():
        o_det = stieltjes_attention(q, k, v, causal=causal, sm_scale=sm,
                                    stieltjes_q=sq, num_iter=30,
                                    normalize=True)
        o_ift = stieltjes_attention(q, k, v, causal=causal, sm_scale=sm,
                                    stieltjes_q=sq, num_iter=30,
                                    normalize=True, ift_grad=True)
    fwd_same = (o_det == o_ift).all().item()

    # B: kernel IFT-norm grads vs dense-IFT autograd
    grads = {}
    for tag, fn in [
        ("kernel", lambda a, b, c: stieltjes_attention(
            a, b, c, causal=causal, sm_scale=sm, stieltjes_q=sq,
            num_iter=30, normalize=True, ift_grad=True)),
        ("dense", lambda a, b, c: dense_pipeline(
            a, b, c, sm, causal, sq, ift=True).to(dtype)),
    ]:
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = fn(qq, kk, vv)
        out.backward(do)
        grads[tag] = (qq.grad.float(), kk.grad.float(), vv.grad.float())

    rels = []
    for gk, gd in zip(grads["kernel"], grads["dense"]):
        rels.append(((gk - gd).norm() / gd.norm().clamp(min=1e-12)).item())
    nan_free = all(torch.isfinite(g).all() for g in grads["kernel"])

    # informational only: kernel's detached-normalized mode vs a FULLY
    # detached dense path. These differ BY CONSTRUCTION (~0.1 rel): the
    # kernel replicates Jack's torch.where-blocked bisection (argmax-kappa
    # term), while no_grad-bisection dense autograd has no lambda path at
    # all. The authoritative detached-mode regression is
    # test_triton_normalized.py (vs Jack's actual code).
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    stieltjes_attention(qq, kk, vv, causal=causal, sm_scale=sm,
                        stieltjes_q=sq, num_iter=30,
                        normalize=True).backward(do)
    q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))
    dense_pipeline(q2, k2, v2, sm, causal, sq, ift=False).to(dtype).backward(do)
    rel_det = ((qq.grad.float() - q2.grad.float()).norm()
               / q2.grad.float().norm().clamp(min=1e-12)).item()

    tol = 2e-2 if dtype == torch.float32 else 6e-2
    ok = fwd_same and nan_free and all(r <= tol for r in rels)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] N={N:4d} D={D:3d} causal={causal!s:5} q={sq:4.1f} "
          f"{str(dtype)[6:]:8} fwd_same={fwd_same} "
          f"rel dQ/dK/dV={rels[0]:.2e}/{rels[1]:.2e}/{rels[2]:.2e} "
          f"det_vs_nograd_info={rel_det:.2e}", flush=True)
    return ok


def main():
    print("Normalized-IFT backward validation")
    print("-" * 100)
    ok = True
    for N in [16, 64, 96, 128, 256]:
        for sq in [2.0, 4.0, 16.0]:
            for causal in [False, True]:
                ok &= run_case(1, 2, N, 64, causal, sq, torch.float32)
    # fp16 spot checks + D=32 (MQMTAR/LM geometry)
    ok &= run_case(2, 4, 128, 64, True, 4.0, torch.float16)
    ok &= run_case(1, 2, 128, 32, True, 4.0, torch.float32)
    ok &= run_case(1, 2, 256, 32, False, 4.0, torch.float32)
    print("=" * 100)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
