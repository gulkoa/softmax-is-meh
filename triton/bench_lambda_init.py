"""Benchmark different initial guesses for the NR solver's λ₀.

After score centering (subtract row max), all scores are ≤ 0 and the maximum
centered score is 0. The NR solver finds λ > 0 such that Σ(λ - xᵢ)^{-q} = 1.

We compare these initial guesses:
  1. n^{1/q}         — exact for uniform scores (current default)
  2. eps              — just above the pole at max(x) = 0 (e.g., 0.1, 1.0)
  3. n^{1/q} / 2      — halved uniform guess
  4. n^{1/q} * 2      — doubled uniform guess

For each, we measure:
  - Iterations to converge (relative error < 1e-6)
  - Final relative error at fixed iteration counts (1, 2, 3, 5)
  - Wall-clock time for the full attention forward pass

Data only — no plotting. Run on compute node.
"""

import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from stieltjes_flash_attn import stieltjes_attention_ref

DEVICE = torch.device("cuda")


def nr_solve(scores, sq, lambda_init, max_iter=20, eps=1e-6):
    """Run NR solver and return convergence history.

    Args:
        scores: (B, H, N, N) — raw (uncentered) scores
        sq: Stieltjes q parameter
        lambda_init: initial guess for λ (after centering)
        max_iter: maximum NR iterations
        eps: clamping epsilon

    Returns:
        list of dicts with iteration, relative_error, f_val
    """
    s_max = scores.max(dim=-1, keepdim=True).values
    x = scores - s_max  # centered, max = 0
    n_cols = scores.shape[-1]

    lambd = torch.full_like(s_max, lambda_init)
    history = []

    # Compute "exact" λ with many iterations
    lambd_exact = torch.full_like(s_max, float(n_cols) ** (1.0 / sq))
    for _ in range(50):
        diff = (lambd_exact - x).clamp(min=eps)
        f_val = diff.pow(-sq).sum(dim=-1, keepdim=True) - 1.0
        f_deriv = -sq * diff.pow(-sq - 1.0).sum(dim=-1, keepdim=True)
        lambd_exact = torch.maximum(lambd_exact - f_val / f_deriv, lambd_exact * 0.5)

    for it in range(max_iter):
        diff = (lambd - x).clamp(min=eps)
        f_val = diff.pow(-sq).sum(dim=-1, keepdim=True) - 1.0
        f_deriv = -sq * diff.pow(-sq - 1.0).sum(dim=-1, keepdim=True)

        rel_err = (lambd - lambd_exact).abs() / lambd_exact.abs().clamp(min=1e-10)
        mean_rel_err = rel_err.mean().item()
        max_rel_err = rel_err.max().item()
        mean_f = f_val.abs().mean().item()

        history.append({
            "iteration": it,
            "mean_rel_error": mean_rel_err,
            "max_rel_error": max_rel_err,
            "mean_abs_f": mean_f,
        })

        lambd = torch.maximum(lambd - f_val / f_deriv, lambd * 0.5)

    return history


def main():
    results_dir = os.environ.get("RESULTS_DIR", ".")
    out_path = os.path.join(results_dir, "lambda_init_comparison.csv")

    torch.manual_seed(42)

    B, H, D = 4, 8, 64
    n_values = [256, 1024, 4096]
    q_values = [1.0, 2.0, 4.0, 8.0]

    fieldnames = [
        "N", "q", "init_name", "init_value", "iteration",
        "mean_rel_error", "max_rel_error", "mean_abs_f",
    ]
    rows = []

    for N in n_values:
        for sq in q_values:
            # Generate random Q, K to get realistic score distributions
            Q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
            K = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
            sm_scale = 1.0 / (D ** 0.5)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale

            uniform_guess = float(N) ** (1.0 / sq)

            init_guesses = {
                "n^(1/q)": uniform_guess,
                "eps=0.1": 0.1,
                "eps=1.0": 1.0,
                "n^(1/q)/2": uniform_guess / 2,
                "n^(1/q)*2": uniform_guess * 2,
                "max+1": 1.0,  # after centering, max=0, so this is 0+1
                "max+n^(1/2q)": float(N) ** (0.5 / sq),
            }

            for init_name, init_val in init_guesses.items():
                print(f"N={N:5d}  q={sq}  init={init_name:15s} (λ₀={init_val:.4f})")
                history = nr_solve(scores, sq, init_val, max_iter=10)

                for h in history:
                    rows.append({
                        "N": N, "q": sq,
                        "init_name": init_name,
                        "init_value": f"{init_val:.6f}",
                        **{k: f"{v:.10f}" for k, v in h.items() if k != "iteration"},
                        "iteration": h["iteration"],
                    })

                # Print convergence summary
                for it in [0, 1, 2, 3]:
                    if it < len(history):
                        h = history[it]
                        print(f"  iter {it}: rel_err={h['mean_rel_error']:.2e}  "
                              f"|f|={h['mean_abs_f']:.2e}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
