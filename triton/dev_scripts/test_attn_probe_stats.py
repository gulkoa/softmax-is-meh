#!/usr/bin/env python3
"""CPU checks for attn_probe_stats (logit-growth probe in train_gpt2_stieltjes).

1. softmax on uniform scores: entropy at query i is ln(i+1) exactly;
2. Stieltjes weights match the repo reference stieltjes_attention_ref
   (weights recovered by passing v = identity), entropy and max |score|.
Run: python test_attn_probe_stats.py
"""
import importlib.util
import math
import sys

import torch

HERE = "/users/PAS2402/alexg/softmax/.worktrees/qknorm/triton/dev_scripts"
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location(
    "trainer", f"{HERE}/train_gpt2_stieltjes.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
from stieltjes_flash_attn import stieltjes_attention_ref  # noqa: E402

torch.manual_seed(0)
B, H, S, D = 1, 3, 64, 16

# 1. uniform softmax
q = torch.zeros(B, H, S, D)
k = torch.randn(B, H, S, D)
st = tr.attn_probe_stats(q, k, "sdpa", 4.0, D)
want = sum(math.log(i + 1) for i in range(S // 2, S)) / (S - S // 2)
assert abs(st["min_head_entropy"] - want) < 1e-5, (st, want)
assert abs(st["mean_head_entropy"] - want) < 1e-5
assert st["max_abs_score"] == 0.0
print(f"uniform softmax entropy ok: {st['min_head_entropy']:.6f} == {want:.6f}")

# 2. Stieltjes vs reference, at several score scales
for scale in (0.5, 2.0, 8.0):
    q = torch.randn(B, H, S, D) * scale
    k = torch.randn(B, H, S, D)
    st = tr.attn_probe_stats(q, k, "stj", 4.0, D)
    eye = torch.eye(S).expand(B, H, S, S).contiguous()
    w = stieltjes_attention_ref(q, k, eye, sm_scale=1 / math.sqrt(D),
                                causal=True, stieltjes_q=4.0, num_iter=30)
    rowsum = w.sum(-1)
    w = w / rowsum[..., None]
    ent = -(w * w.clamp_min(1e-30).log()).sum(-1)[..., S // 2:].mean(dim=(0, 2))
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    mx = sc.masked_fill(~torch.ones(S, S, dtype=torch.bool).tril(), 0).abs().max().item()
    assert abs(st["min_head_entropy"] - ent.min().item()) < 1e-4, (scale, st, ent)
    assert abs(st["mean_head_entropy"] - ent.mean().item()) < 1e-4
    assert abs(st["max_abs_score"] - mx) < 1e-4 * max(1.0, mx)
    print(f"stj scale {scale}: entropy {st['min_head_entropy']:.4f}/"
          f"{st['mean_head_entropy']:.4f} (ref {ent.min():.4f}/{ent.mean():.4f}), "
          f"max|s| {st['max_abs_score']:.3f}, ref row sums {rowsum.min():.5f}-{rowsum.max():.5f}")

# 3. softmax entropy falls as scores sharpen (sanity)
e = [tr.attn_probe_stats(torch.randn(B, H, S, D) * c, torch.randn(B, H, S, D),
                         "sdpa", 4.0, D)["mean_head_entropy"] for c in (0.1, 1, 10)]
assert e[0] > e[1] > e[2], e
print("softmax entropy decreases with score scale:", [round(x, 3) for x in e])
print("ALL OK")
