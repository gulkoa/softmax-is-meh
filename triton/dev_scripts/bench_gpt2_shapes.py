"""M0 for the GPT-2 side project: fwd+bwd attention throughput at
GPT-2-small shapes (B x 12H x 1024 x 64), Stieltjes (NR-8 / Halley-3,
normalize + ift_grad) vs SDPA, bf16 causal — grounds the token budget."""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402


def bench(fn, q, k, v, do, iters=10):
    for _ in range(3):
        out = fn(q, k, v)
        out.backward(do)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        out = fn(q, k, v)
        out.backward(do)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    B, H, N, D = 16, 12, 1024, 64          # GPT-2 small, bs 16k tokens
    q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16,
                    requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    do = torch.randn_like(q)
    sm = 0.125

    arms = {
        "sdpa": lambda a, b, c: F.scaled_dot_product_attention(
            a, b, c, is_causal=True),
        "stj-nr8": lambda a, b, c: stieltjes_attention(
            a, b, c, causal=True, sm_scale=sm, stieltjes_q=4.0,
            num_iter=8, normalize=True, ift_grad=True),
        "stj-hal3": lambda a, b, c: stieltjes_attention(
            a, b, c, causal=True, sm_scale=sm, stieltjes_q=4.0,
            num_iter=3, solver="halley", normalize=True, ift_grad=True),
    }
    tokens = B * N
    base = None
    for tag, fn in arms.items():
        ms = bench(fn, q, k, v, do)
        if tag == "sdpa":
            base = ms
        # 12 layers of attention per model step; attention share of a GPT-2
        # step is ~35-45% for sdpa — report attention-only numbers plus a
        # rough model-step extrapolation assuming non-attention cost equals
        # the sdpa-model's other 60% (measured separately in M1)
        print(f"{tag:9s}: {ms:8.2f} ms/layer-call  "
              f"(x{ms/base:5.2f} vs sdpa)  "
              f"12-layer attn: {12*ms:8.1f} ms per {tokens} tokens "
              f"-> {tokens/(12*ms/1000)/1e3:8.1f}k tok/s attn-bound",
              flush=True)


if __name__ == "__main__":
    main()
