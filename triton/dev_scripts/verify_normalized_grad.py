"""
Verify the analytical gradient of the NORMALIZED Stieltjes (the new `stieltjes`
reference) against PyTorch autograd, so it can be implemented in the Triton
backward.

Normalized Stieltjes (lambda detached, max-centered):
  shifted_j = s_j - s_max            (s_max = max, DIFFERENTIABLE -> argmax term)
  w_j = (lam - shifted_j)^(-q)       (lam detached)
  r_j = (lam - shifted_j)^(-q-1)
  S = sum_j w_j,  R = sum_j r_j,  p_j = w_j / S
  O = sum_j p_j v_j

Given dO, with dP_j = dO . v_j:
  A = sum_j dP_j r_j
  B = sum_j dP_j p_j
Claim:
  dS_k = (q/S) [ r_k (dP_k - B) - [k==argmax] (A - R*B) ]

We check dS_k against torch.autograd of O w.r.t. s.
"""
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def analytical_dS(s, lam, v, dO, q):
    # s: (T,), lam: scalar (detached), v: (T, D), dO: (D,)
    s_max, argmax = s.max(dim=0)
    shifted = s - s_max
    diff = (lam - shifted).clamp(min=1e-12)
    w = diff.pow(-q)
    r = diff.pow(-q - 1.0)
    S = w.sum()
    R = r.sum()
    p = w / S
    dP = (v * dO).sum(dim=1)            # (T,)  dP_j = dO . v_j
    A = (dP * r).sum()
    B = (dP * p).sum()
    dS = (q / S) * (r * (dP - B))
    # argmax term: subtract (q/S)(A - R B) at k == argmax
    dS = dS.clone()
    dS[argmax] = dS[argmax] - (q / S) * (A - R * B)
    return dS


def autograd_dS(s, lam, v, dO, q):
    s = s.clone().requires_grad_(True)
    s_max = s.max()
    shifted = s - s_max
    diff = (lam - shifted).clamp(min=1e-12)
    w = diff.pow(-q)
    p = w / w.sum()
    O = (p.unsqueeze(1) * v).sum(dim=0)   # (D,)
    O.backward(dO)
    return s.grad


def main():
    print("Verify normalized-Stieltjes analytical gradient vs autograd (fp64)\n")
    print(f"{'q':>4} {'T':>4} | {'max|ana-autograd|':>18} {'rel':>10}")
    for q in [1.0, 2.0, 4.0, 8.0, 16.0]:
        for T in [8, 16, 64]:
            s = torch.randn(T) * 1.5
            v = torch.randn(T, 16)
            dO = torch.randn(16)
            # solve lam by bisection on shifted so S ~ 1 (detached)
            s_max = s.max()
            shifted = s - s_max
            lb = torch.tensor(1e-9)
            ub = torch.tensor(float(T) ** (1.0 / q))
            for _ in range(60):
                mid = (lb + ub) / 2
                f = (mid - shifted).clamp(min=1e-12).pow(-q).sum() - 1.0
                lb = torch.where(f > 0, mid, lb)
                ub = torch.where(f <= 0, mid, ub)
            lam = ((lb + ub) / 2).detach()

            ana = analytical_dS(s, lam, v, dO, q)
            ag = autograd_dS(s, lam, v, dO, q)
            err = (ana - ag).abs().max().item()
            rel = err / (ag.abs().max().item() + 1e-30)
            flag = "" if rel < 1e-6 else "  <-- MISMATCH"
            print(f"{q:>4.0f} {T:>4} | {err:>18.3e} {rel:>10.3e}{flag}")
    print("\nIf all rel < 1e-6, the analytical formula is correct and ready for Triton.")


if __name__ == "__main__":
    main()
