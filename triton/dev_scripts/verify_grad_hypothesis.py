"""
Verify the gradient-flow hypothesis from the advisor:
- BS uses torch.where(prob_sum > 0, mid, lb) → λ has no gradient w.r.t. x.
- NR uses iterative update → λ has gradient w.r.t. x via implicit function thm.

If true, dP/dx_i from autograd should differ substantially between BS and
NR even when the forward outputs agree to ~1e-5 (e.g. q=4).

We also compute the "ground truth" dP/dx_i using the implicit function
theorem analytically:
  P_i  = (λ - x_i)^{-q}
  dλ/dx_i = (λ - x_i)^{-q-1} / Σ_j (λ - x_j)^{-q-1}
  dP_i/dx_k = q (λ - x_i)^{-q-1} (δ_{ik} - dλ/dx_k)
  i.e. dP_i/dx_i = q (λ-x_i)^{-q-1} (1 - r_i / Σ r_j) where r_j = (λ-x_j)^{-q-1}.
"""
import sys
sys.path.insert(0, "/users/PAS2402/alexg/softmax/tmp")

import torch
from maxretr_bs_vs_nr import StieltjesBSTransform, StieltjesNRTransform


def implicit_thm_jacobian(x: torch.Tensor, lam: torch.Tensor, q: float):
    """Analytic dP/dx via implicit function theorem (treating λ as defined
    implicitly by Σ(λ-x_j)^(-q) = 1)."""
    # x: (..., N), lam: (..., 1)
    diff = (lam - x)
    r = diff.pow(-q - 1.0)        # (..., N)
    R = r.sum(dim=-1, keepdim=True)  # (..., 1)
    P = diff.pow(-q)              # (..., N)

    # dP_i/dx_k = q * (lam-x_i)^{-q-1} * (delta_{ik} - r_k / R)
    # Jacobian shape: (..., N, N) where row=i, col=k
    delta = torch.eye(x.shape[-1], device=x.device).expand(*x.shape[:-1], -1, -1)
    # r_k / R: (..., 1, N)
    rR = (r / R).unsqueeze(-2)
    Ji_part = (delta - rR)        # (..., N, N), broadcast
    # multiply by q * r_i^* but r_i = (lam-x_i)^{-q-1}, same as r
    Jacobian = q * r.unsqueeze(-1) * Ji_part
    return Jacobian, P


def autograd_jacobian(mapping, x: torch.Tensor, q: float):
    """Compute dP/dx using torch.autograd.functional.jacobian."""
    def f(xx):
        return mapping.translate_logits(xx, dim=-1)
    return torch.autograd.functional.jacobian(f, x, vectorize=False)


def main():
    torch.manual_seed(0)
    device = "cpu"

    # Simple input: one row, T=16 keys
    x = torch.randn(16, device=device, dtype=torch.float64) * 0.5  # mild range
    print(f"x = {x.tolist()}")

    for q in [1.0, 4.0, 16.0]:
        print(f"\n{'='*72}\n q = {q}\n{'='*72}")

        bs = StieltjesBSTransform(q=q, num_iter=64, eps=1e-12).to(torch.float64)
        nr = StieltjesNRTransform(q=q, num_iter=20, eps=1e-12).to(torch.float64)
        # bumped iterations so both are well-converged (forward equivalence)

        # Forward
        w_bs = bs.translate_logits(x.unsqueeze(0), dim=-1).squeeze(0)
        w_nr = nr.translate_logits(x.unsqueeze(0), dim=-1).squeeze(0)
        print(f"forward max|Δ| = {(w_bs - w_nr).abs().max().item():.3e}")
        print(f"BS row sum = {w_bs.sum().item():.8f}")
        print(f"NR row sum = {w_nr.sum().item():.8f}")

        # Get λ via NR for analytic IFT
        s_max = x.max()
        x_c = x - s_max
        lam_nr = torch.tensor(1.1, dtype=torch.float64)
        for _ in range(50):
            diff = (lam_nr - x_c).clamp(min=1e-12)
            f_val = diff.pow(-q).sum() - 1.0
            f_deriv = -q * diff.pow(-q - 1.0).sum()
            lam_nr = lam_nr - f_val / f_deriv
        J_analytic, P_check = implicit_thm_jacobian(
            x_c, lam_nr.unsqueeze(0), q)
        print(f"converged lam = {lam_nr.item():.8f}, "
              f"P check row sum = {P_check.sum().item():.8f}")

        # Autograd jacobians
        xa = x.clone().requires_grad_(True)
        J_bs = autograd_jacobian(bs, xa.unsqueeze(0), q).squeeze()
        J_nr = autograd_jacobian(nr, xa.unsqueeze(0), q).squeeze()

        J_analytic = J_analytic.squeeze()  # (16, 16)

        print(f"\nJacobian shapes: BS {J_bs.shape}, NR {J_nr.shape}, analytic {J_analytic.shape}")
        print(f"BS Jacobian   diag mean = {J_bs.diag().mean().item():.6e}, "
              f"off-diag mean = {(J_bs - torch.diag(J_bs.diag())).mean().item():.6e}")
        print(f"NR Jacobian   diag mean = {J_nr.diag().mean().item():.6e}, "
              f"off-diag mean = {(J_nr - torch.diag(J_nr.diag())).mean().item():.6e}")
        print(f"Ana Jacobian  diag mean = {J_analytic.diag().mean().item():.6e}, "
              f"off-diag mean = {(J_analytic - torch.diag(J_analytic.diag())).mean().item():.6e}")

        print(f"\nmax|BS_J - NR_J|  = {(J_bs - J_nr).abs().max().item():.6e}")
        print(f"max|NR_J - Ana_J| = {(J_nr - J_analytic).abs().max().item():.6e}")
        print(f"max|BS_J - Ana_J| = {(J_bs - J_analytic).abs().max().item():.6e}")

        # The "wrong" BS Jacobian should be the naive  q (λ - x_i)^{-q-1} δ_{ij}
        # (i.e. treating λ as constant)
        diff = (lam_nr - x_c)
        r = diff.pow(-q - 1.0)
        J_naive = q * torch.diag(r)
        print(f"max|BS_J - naive_diag| = {(J_bs - J_naive).abs().max().item():.6e}")


if __name__ == "__main__":
    main()
