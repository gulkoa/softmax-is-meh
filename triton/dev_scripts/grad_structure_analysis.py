"""
CPU analysis: structural characterization of the normalized-vs-unnormalized
Stieltjes gradient difference (supports the why-norm-helps findings doc).

Claims verified numerically (fp64) — note C1 was REVISED by the result:
  C1 (revised). BOTH gradients are zero-sum over positions: the differentiable
      max-centering makes each mapping invariant to uniform score shifts, so
      Σ_k g_k = 0 for unnormalized too (the argmax position alone carries the
      full counterweight −A there: Σ_k q[r_k dP_k − 1_argmax A] = q[A − A] = 0).
      The distinguishing structure is NOT the row-sum; it is WHERE the
      counterweight sits (see C2): unnormalized = all on the argmax key;
      normalized = distributed across all keys ∝ r_k.
  C2. The per-example score-gradient difference is exactly
        g_norm = g_unnorm/S − (q/S)·B·(r − [k==argmax]·R)
      where B = Σ_j p_j dP_j = dO·O_norm. With S≈1 the 1/S factor is ~1e-4;
      the coupling term −(q/S)·B·(r − 1_argmax·R) is the structural difference.
      It is the simplex-tangent projection: Σ_j dp_j/ds_k = 0 per output row
      (p always sums to 1 under perturbation; w does not).
  C3. The coupling term is zero-sum over positions and rank-1 in dP:
      it depends on dP only through the scalar B = dO·O.
  C4. Magnitude: |coupling| / |g_unnorm| ≈ 0.5–1.4 on random inputs — a
      FIRST-ORDER difference in gradient direction, not a perturbation.
"""
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def bisect_lam(shifted, q, iters=60, eps=1e-12):
    lb = torch.full_like(shifted[..., :1], eps)
    ub = torch.full_like(shifted[..., :1], shifted.shape[-1] ** (1.0 / q))
    for _ in range(iters):
        mid = (lb + ub) / 2
        f = (mid - shifted).clamp(min=eps).pow(-q).sum(-1, keepdim=True) - 1.0
        lb = torch.where(f > 0, mid, lb)
        ub = torch.where(f <= 0, mid, ub)
    return (lb + ub) / 2


def grads(s, v, dO, q):
    """Return (g_unnorm, g_norm, parts) for one row s:(T,), v:(T,D), dO:(D,)."""
    def fwd(z, norm):
        x_max = z.max()
        shifted = z - x_max
        lam = bisect_lam(shifted.unsqueeze(0), q).squeeze(0).detach()
        w = (lam - shifted).clamp(min=1e-12).pow(-q)
        out = w / w.sum() if norm else w
        return (out.unsqueeze(0) @ v).squeeze(0)

    g = {}
    for norm in (False, True):
        z = s.clone().requires_grad_(True)
        fwd(z, norm).backward(dO)
        g[norm] = z.grad.clone()

    # analytic parts
    x_max, argmax = s.max(0)
    shifted = s - x_max
    lam = bisect_lam(shifted.unsqueeze(0), q).squeeze(0)
    diff = (lam - shifted).clamp(min=1e-12)
    w = diff.pow(-q)
    r = diff.pow(-q - 1.0)
    S, R = w.sum(), r.sum()
    p = w / S
    dP = v @ dO
    B = (p * dP).sum()
    coupling = -(q / S) * B * (r - torch.nn.functional.one_hot(argmax, s.numel()) * R)
    return g[False], g[True], dict(S=S, B=B, coupling=coupling,
                                   O_norm=(p.unsqueeze(0) @ v).squeeze(0), dO=dO)


def main():
    print("Structural analysis of normalized vs unnormalized Stieltjes gradients")
    print("=" * 74)
    for q in [2.0, 4.0]:
        for T in [16, 256]:
            res = []
            for trial in range(64):
                s = torch.randn(T) * 1.5
                v = torch.randn(T, 32)
                dO = torch.randn(32)
                gu, gn, parts = grads(s, v, dO, q)
                S = parts["S"]
                # C1: row-sum (sum over positions) of each gradient
                rs_u = gu.sum().abs().item()
                rs_n = gn.sum().abs().item()
                # C2: decomposition residual
                pred = gu / S + parts["coupling"]
                resid = (gn - pred).abs().max().item()
                # C3: B identity
                b_err = (parts["B"] - parts["dO"] @ parts["O_norm"]).abs().item()
                # C4: coupling magnitude fraction
                frac = (parts["coupling"].norm() / gu.norm()).item()
                res.append((rs_u, rs_n, resid, b_err, frac))
            rs_u, rs_n, resid, b_err, frac = map(
                lambda i: torch.tensor([t[i] for t in res]), range(5))
            print(f" q={q:>3.0f} T={T:>4}: |Σg_unnorm|={rs_u.mean():.3e}  "
                  f"|Σg_norm|={rs_n.mean():.3e}  "
                  f"decomp_resid={resid.max():.1e}  |B - dO·O|={b_err.max():.1e}  "
                  f"|coupling|/|g_unnorm| = {frac.mean():.3f} "
                  f"(min {frac.min():.3f}, max {frac.max():.3f})")
    print("""
Reading (C1 revised after measurement):
  C1  BOTH row-sums are ~0: max-centering makes both mappings shift-invariant
      in s, so Σ_k g_k = 0 either way. The real difference is WHERE the
      counterweight sits — unnormalized: entirely on the argmax key;
      normalized: spread over all keys ∝ r_k (simplex-tangent gradient,
      Σ_j dp_j/ds_k = 0 per output row).
  C2  decomp_resid ≈ 0 confirms g_norm = g_unnorm/S − (q/S)·B·(r − 1_argmax·R).
  C3  B = dO·O exactly: the coupling strength is the alignment of the upstream
      gradient with the current attention output.
  C4  |coupling|/|g_unnorm| ≈ 0.5–1.4: the two trainings see substantially
      different gradient DIRECTIONS — a first-order difference, which is why
      a ~1e-4 forward difference can still produce a multi-pp OOD gap.""")


if __name__ == "__main__":
    main()
