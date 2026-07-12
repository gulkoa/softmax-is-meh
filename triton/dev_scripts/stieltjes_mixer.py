"""
Zoology sequence-mixer wrapping the Triton Stieltjes flash kernel.

Drop-in replacement for zoology.mixers.attention.MHA: same Wqkv/out_proj
structure, but the inner attention is stieltjes_attention (causal,
normalize=True by default — matches Jack's normalized `stieltjes`).

Routes around the still-undelivered copy/MQMTR task code by using zoology's
MQAR harness directly (the June-2 meeting's designated testbed family).

Notes:
  - head_dim = d_model / num_heads must be in {16, 32, 64, 128, 256}.
  - The kernel runs in bf16 internally (params/IO stay fp32): fp32 would use
    the slow IEEE matmul path; bf16 is validated for q <= 4 (job 12312766).
  - `dropout` is applied to the attention OUTPUT (the fused kernel cannot
    drop attention probabilities). For fair comparisons set dropout equal
    (ideally 0.0) in both the softmax and Stieltjes arms.
"""
import math
import sys

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402


class StieltjesMHA(nn.Module):
    """Multi-head causal self-attention with Stieltjes normalization."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: int = None,
        stieltjes_q: float = 4.0,
        num_iter: int = 8,
        normalize: bool = True,
        block_lambda_grad: bool = False,
        compute_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        assert self.head_dim in {16, 32, 64, 128, 256}, (
            f"head_dim={self.head_dim} unsupported by the Triton kernel")
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.stieltjes_q = float(stieltjes_q)
        self.num_iter = int(num_iter)
        self.normalize = bool(normalize)
        self.block_lambda_grad = bool(block_lambda_grad)
        self.compute_dtype = {"bf16": torch.bfloat16,
                              "fp16": torch.float16,
                              "fp32": torch.float32}[compute_dtype]

    def forward(self, x: torch.Tensor):
        # x: (B, S, d_model)
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d",
                        three=3, d=self.head_dim)
        q, k, v = qkv.unbind(dim=2)                     # (B, S, H, D)
        q = rearrange(q, "b s h d -> b h s d").contiguous()
        k = rearrange(k, "b s h d -> b h s d").contiguous()
        v = rearrange(v, "b s h d -> b h s d").contiguous()

        in_dtype = q.dtype
        cd = self.compute_dtype
        o = stieltjes_attention(
            q.to(cd), k.to(cd), v.to(cd),
            causal=True,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            stieltjes_q=self.stieltjes_q,
            num_iter=self.num_iter,
            block_lambda_grad=self.block_lambda_grad,
            normalize=self.normalize,
        ).to(in_dtype)                                   # (B, H, S, D)

        o = rearrange(o, "b h s d -> b s (h d)")
        o = F.dropout(o, self.dropout_p if self.training else 0.0)
        return self.out_proj(o)

    def state_size(self, batch_size: int = 1, sequence_length: int = 2048):
        return 2 * self.d_model * sequence_length
