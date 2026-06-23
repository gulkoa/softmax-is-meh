"""
Debug why probability-simplex-mappings/mappings/stieltjes.py bisection
fails to converge at q=1, q=16, q=32.

Hypothesis A: insufficient iterations — bracket is correct, just need more.
Hypothesis B: incorrect upper bound — ub = N^(1/q) is not actually a valid
  upper bound for f(λ) = Σ(λ - x_i)^(-q) - 1.
Hypothesis C: numerical artifact of pow(diff, -q) for small diff at large q.

Method: replicate the bisection exactly, instrument it, and inspect what
happens at each iteration on the same fixed-input batch the smoke test used.
"""

from __future__ import annotations

import sys

import torch

PSM_ROOT = "/users/PAS2402/alexg/softmax/probability-simplex-mappings"
if PSM_ROOT not in sys.path:
    sys.path.insert(0, PSM_ROOT)


def bisect_instrumented(scores: torch.Tensor, q: float, num_iter: int,
                        eps: float, dim: int = -1):
    """Reproduces mappings/stieltjes.py @ a19bb33 with per-iter logging.

    Returns:
      lambdas:  list of λ at each iter (mean across rows)
      sums:     list of row-sum at each iter (mean)
      sum_min:  list of min row-sum (worst-converged row)
      sum_max:  list of max row-sum
    """
    scores = torch.clamp(scores, min=-50.0, max=50.0)
    x_max = torch.max(scores, dim=dim, keepdim=True).values
    x_i = scores - x_max  # ≤ 0

    N = scores.shape[dim]
    lb = torch.full_like(x_max, eps)
    ub = torch.full_like(x_max, N ** (1.0 / q))

    print(f"  Initial: lb=eps={eps}  ub=N^(1/q)={N**(1/q):.4e}")

    # Compute f at the initial bracket
    def f(lam):
        return (torch.pow((lam - x_i).clamp(min=eps), -q)
                .sum(dim=dim, keepdim=True) - 1.0)

    f_lb = f(lb)
    f_ub = f(ub)
    print(f"  f(lb)  mean={f_lb.mean().item():.4e}  min={f_lb.min().item():.4e}  max={f_lb.max().item():.4e}")
    print(f"  f(ub)  mean={f_ub.mean().item():.4e}  min={f_ub.min().item():.4e}  max={f_ub.max().item():.4e}")
    print("  (f(lb) should be POSITIVE; f(ub) should be NEGATIVE for valid bracket)")

    lambdas, sums, sum_min, sum_max = [], [], [], []
    for it in range(num_iter):
        mid = (lb + ub) / 2.0
        prob_sum_minus_1 = f(mid)
        row_sum = prob_sum_minus_1 + 1.0

        lb = torch.where(prob_sum_minus_1 > 0, mid, lb)
        ub = torch.where(prob_sum_minus_1 <= 0, mid, ub)

        lambdas.append(mid.mean().item())
        sums.append(row_sum.mean().item())
        sum_min.append(row_sum.min().item())
        sum_max.append(row_sum.max().item())

    return lambdas, sums, sum_min, sum_max


def newton_instrumented(scores: torch.Tensor, q: float, num_iter: int,
                        eps: float, lam_init: float = 1.1, dim: int = -1):
    """Newton-Raphson loop matching softmax-is-meh/triton/stieltjes_flash_attn.py."""
    s_max = scores.max(dim=dim, keepdim=True).values
    x = scores - s_max
    lambd = torch.full_like(s_max, lam_init)

    lambdas, sums = [], []
    for _ in range(num_iter):
        diff = (lambd - x).clamp(min=eps)
        f_val = diff.pow(-q).sum(dim=dim, keepdim=True) - 1.0
        f_deriv = -q * diff.pow(-q - 1.0).sum(dim=dim, keepdim=True)
        lambd = lambd - f_val / f_deriv
        diff = (lambd - x).clamp(min=eps)
        row_sum = diff.pow(-q).sum(dim=dim, keepdim=True)
        lambdas.append(lambd.mean().item())
        sums.append(row_sum.mean().item())
    return lambdas, sums


def main():
    torch.manual_seed(0)
    device = "cpu"
    B, T, D = 4, 16, 32
    queries = torch.randn(B, 1, D, device=device)
    keys = torch.randn(B, T, D, device=device)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * (D ** -0.5)

    print(f"Input: scores shape {tuple(scores.shape)}, dtype {scores.dtype}")
    print(f"  scores range [{scores.min().item():.3f}, {scores.max().item():.3f}]")

    for q in [1.0, 4.0, 8.0, 16.0]:
        print(f"\n{'='*72}\n q = {q}\n{'='*72}")
        print("\nBisection (16 iters, eps=1e-9):")
        lams_bs, sums_bs, smin, smax = bisect_instrumented(scores, q, num_iter=16, eps=1e-9)
        # Show every 4th iteration plus last few
        for i, (lam, s, lo, hi) in enumerate(zip(lams_bs, sums_bs, smin, smax)):
            if i % 4 == 0 or i >= 14:
                print(f"  iter {i:2d}: λ_mean={lam:.4e}  row_sum mean={s:.4e}  "
                      f"min={lo:.4e}  max={hi:.4e}")

        print("\nNewton-Raphson (5 iters, init=1.1, eps=1e-6):")
        lams_nr, sums_nr = newton_instrumented(scores, q, num_iter=5, eps=1e-6)
        for i, (lam, s) in enumerate(zip(lams_nr, sums_nr)):
            print(f"  iter {i:2d}: λ_mean={lam:.4e}  row_sum mean={s:.4e}")

        # What does bisection do with MORE iterations?
        print("\nBisection with 64 iters:")
        lams64, sums64, smin64, smax64 = bisect_instrumented(
            scores, q, num_iter=64, eps=1e-9)
        # Just final iter
        print(f"  final (iter 64): λ_mean={lams64[-1]:.4e}  row_sum mean={sums64[-1]:.4e}  "
              f"min={smin64[-1]:.4e}  max={smax64[-1]:.4e}")

        # And with smaller eps?
        print("\nBisection with 16 iters, eps=1e-30:")
        lams_e, sums_e, smin_e, smax_e = bisect_instrumented(
            scores, q, num_iter=16, eps=1e-30)
        print(f"  final: λ_mean={lams_e[-1]:.4e}  row_sum mean={sums_e[-1]:.4e}  "
              f"min={smin_e[-1]:.4e}  max={smax_e[-1]:.4e}")


if __name__ == "__main__":
    main()
