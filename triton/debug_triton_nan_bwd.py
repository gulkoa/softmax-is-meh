"""
Toy reproducer for --stieltjes-use-triton NaN in training.

Configs where NaN might appear:
 - correlated QKV (Q=K from shared projection of x)
 - bigger lr (so weights actually change)
 - larger N (so score distribution tails matter)
 - run enough steps for weights to drift
"""
import argparse
import torch
import torch.nn as nn
from stieltjes_flash_attn import StieltjesAttention, stieltjes_attention_ref


def has_nan(x):
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


class MiniAttn(nn.Module):
    """Mini 1-head self-attn: x -> qkv_proj -> stj_attention -> o_proj."""
    def __init__(self, D, H, backend, q, num_iter, causal):
        super().__init__()
        self.H = H
        self.dh = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.backend = backend
        self.q = q
        self.num_iter = num_iter
        self.causal = causal
        self.sm_scale = 1.0 / (self.dh ** 0.5)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # (B,H,N,dh)
        if self.backend == "triton":
            o = StieltjesAttention.apply(Q, K, V, self.causal, self.sm_scale,
                                         self.q, self.num_iter)
        else:
            o = stieltjes_attention_ref(Q, K, V, self.sm_scale, causal=self.causal,
                                         stieltjes_q=self.q, num_iter=self.num_iter)
        o = o.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.o(o)


def run(backend, B, N, D, H, q, num_iter, causal, steps, lr, dtype, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    model = MiniAttn(D, H, backend, q, num_iter, causal).to(dev).to(dtype)
    x = torch.randn(B, N, D, device=dev, dtype=dtype)
    target = torch.randn(B, N, D, device=dev, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[{backend}] B={B} N={N} D={D} H={H} q={q} num_iter={num_iter} "
          f"causal={causal} dtype={dtype} steps={steps} lr={lr}")
    for step in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        # Collect max grad magnitude and nan flag across params
        max_grad = 0.0
        any_nan = False
        for name, p in model.named_parameters():
            if p.grad is None: continue
            if has_nan(p.grad):
                any_nan = True
                print(f"    NaN in grad of {name}")
            else:
                max_grad = max(max_grad, p.grad.abs().max().item())
        out_nan = has_nan(out)
        loss_nan = has_nan(loss)
        tag = "OK" if not (any_nan or out_nan or loss_nan) else "NaN"
        print(f"  step {step}: loss={loss.item():.4f} |out|max={out.detach().abs().max().item():.3g} "
              f"|grad|max={max_grad:.3g}  {tag}")
        if any_nan or out_nan or loss_nan:
            break
        opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--D", type=int, default=384)
    p.add_argument("--H", type=int, default=6)
    p.add_argument("--q", type=float, default=4.0)
    p.add_argument("--num-iter", type=int, default=3)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="both", choices=["triton", "ref", "both"])
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    if args.backend in ("ref", "both"):
        run("ref", args.B, args.N, args.D, args.H, args.q, args.num_iter,
            args.causal, args.steps, args.lr, dtype, args.seed)
        print()
    if args.backend in ("triton", "both"):
        run("triton", args.B, args.N, args.D, args.H, args.q, args.num_iter,
            args.causal, args.steps, args.lr, dtype, args.seed)


if __name__ == "__main__":
    main()
