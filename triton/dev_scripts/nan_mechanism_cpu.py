"""
Simplest possible demonstration of the NaN mechanism — pure CPU arithmetic.

This replicates EXACTLY what the Triton backward dkdv/dq kernels compute for a
PADDED query row (a row index offs_m >= N_CTX that exists only because
BLOCK_M > N_CTX).

Key asymmetry vs the forward kernel:
  - Forward: for a padded query row, lambda is RECOMPUTED via NR starting from
    init=1.1, so it converges to a normal value (~1.x). Weights are normal.
  - Backward: lam_row is LOADED from the saved Lambda tensor with `other=0.0`
    for padded rows (they were never stored). So lam_row = 0.

We trace the arithmetic for a padded row in the backward and show where NaN
appears, for each q and each dtype.
"""
import torch

EPS = 1e-6  # matches the kernel's EPS

print("=" * 76)
print("Padded-row backward arithmetic (what the Triton bwd kernel computes)")
print("=" * 76)
print()
print("For a PADDED query row (offs_m >= N_CTX):")
print("  q_block (this query row)  = 0      (tl.load ..., other=0.0)")
print("  qk = q_block @ k^T        = 0")
print("  lam_row                   = 0      (tl.load Lambda ..., other=0.0)")
print("  diff = max(lam_row - qk, EPS) = max(0 - 0, EPS) = EPS = 1e-6")
print("  do_block (this row)       = 0      (tl.load ..., other=0.0)")
print()

for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
    print("-" * 76)
    print(f"dtype for the matmul cast = {dtype_name}")
    print("-" * 76)
    print(f"{'q':>4} | {'weights=EPS^-q':>16} | {'r=EPS^(-q-1)':>16} | "
          f"{'w.to(dtype)':>12} | {'dV term=w*0':>12} | {'dK term=r*0':>12}")
    for q in [1.0, 4.0, 16.0]:
        # All intermediate math in fp32 (kernel uses tl.float32 internally)
        diff = torch.tensor(EPS, dtype=torch.float32)
        weights = diff.pow(-q)          # (lam - x)^-q
        r = diff.pow(-q - 1.0)          # (lam - x)^(-q-1)

        # dV accumulation: dv += dot(trans(weights.to(do.dtype)), do_block)
        #   weights cast to the V/dO dtype, then multiplied by do_block (=0 here)
        w_cast = weights.to(dtype)
        dV_term = w_cast.to(torch.float32) * torch.tensor(0.0)  # inf*0 if w_cast=inf

        # dK accumulation: dS = sq*r*(dP - delta); dP=0, delta=0 for padded → r*0
        # r stays fp32 in the kernel; cast happens after. Show r*0 in fp32.
        dK_term = (q * r) * torch.tensor(0.0)  # inf*0 if r=inf

        print(f"{q:>4.0f} | {weights.item():>16.3e} | {r.item():>16.3e} | "
              f"{w_cast.item():>12.3e} | {dV_term.item():>12} | {dK_term.item():>12}")
    print()

print("=" * 76)
print("INTERPRETATION")
print("=" * 76)
print("""
fp16 max finite value = 65504. fp32 max finite value = 3.4e38.

dV term = weights.to(matmul_dtype) * do_block[padded]=0:
  - fp16 cast: EPS^-1 = 1e6 > 65504 -> inf at EVERY q -> inf*0 = NaN at every q.
  - fp32:      EPS^-1=1e6, EPS^-4=1e24 finite -> *0 = 0 (no NaN);
               EPS^-16=1e96 > 3.4e38 -> inf -> NaN only at q=16.

dK term = q*r*0 where r=EPS^(-q-1), computed in fp32:
  - q=1:  r=1e12 finite -> *0=0 (no NaN)
  - q=4:  r=1e30 finite -> *0=0 (no NaN)
  - q=16: r=1e102 > 3.4e38 -> inf -> NaN

dQ: the dq kernel masks padded query rows on STORE and its inner loop is over
    K-tiles (N=64=BLOCK_N, no K padding), so valid rows are never contaminated.
    -> dQ never NaN.

This exactly matches the observed pattern in check-triton-blg-10446632.out:
  dV NaN at all q; dK NaN only at q=16; dQ never NaN; fp16 makes dV worse.
""")
