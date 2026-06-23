"""Direct trace of StieltjesTransform.translate_logits at q=16."""
import sys
sys.path.insert(0, "/users/PAS2402/alexg/softmax/probability-simplex-mappings")

import torch
from mappings.stieltjes import StieltjesTransform

torch.manual_seed(0)
B, T, D = 4, 16, 32
queries = torch.randn(B, 1, D)
keys = torch.randn(B, T, D)
scores = torch.matmul(queries, keys.transpose(-2, -1)) * (D ** -0.5)

q, num_iter, eps = 16.0, 16, 1e-9

# Inline the StieltjesTransform.translate_logits, with tracing

logits = torch.clamp(scores, min=-50.0, max=50.0)
x_max = torch.max(logits, dim=-1, keepdim=True).values
x_i = logits - x_max  # shape (4, 1, 16), values ≤ 0
print("x_i shape:", x_i.shape)
print("x_i row 0:", x_i[0, 0].tolist())

N = logits.shape[-1]
lb = torch.full_like(x_max, eps)
ub = torch.full_like(x_max, N ** (1.0 / q))
print(f"Init: lb={eps}, ub=N^(1/q)={N ** (1.0/q):.6f}")

for it in range(num_iter):
    mid = (lb + ub) / 2.0
    diff = (mid - x_i).clamp(min=eps)
    prob_sum = torch.pow(diff, -q).sum(dim=-1, keepdim=True) - 1.0
    if it < 4 or it >= num_iter - 4:
        rs = prob_sum + 1.0
        print(f"  iter {it:2d}: mid={mid.flatten().tolist()}  "
              f"row_sum={rs.flatten().tolist()}")
    lb = torch.where(prob_sum > 0, mid, lb)
    ub = torch.where(prob_sum <= 0, mid, ub)

lambda_1 = (lb + ub) / 2.0
print(f"\nFinal lambda_1: {lambda_1.flatten().tolist()}")

# Now compute weights
diff = (lambda_1 - x_i).clamp(min=eps)
weights = torch.pow(diff, -q)
print(f"Final row sums: {weights.sum(dim=-1).flatten().tolist()}")

# Compare with the package's call
print("\n--- Package call (StieltjesTransform.translate_logits) ---")
bs = StieltjesTransform(q=q, num_iter=num_iter, eps=eps)
w_pkg = bs.translate_logits(scores, dim=-1)
print(f"Package row sums: {w_pkg.sum(dim=-1).flatten().tolist()}")
