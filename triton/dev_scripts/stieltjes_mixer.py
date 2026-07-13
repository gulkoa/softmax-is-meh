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


class SdpaMHA(nn.Module):
    """Softmax MHA via F.scaled_dot_product_attention (flash) — memory-linear
    baseline for hard-stretch evals where zoology's dense MHA would OOM."""

    def __init__(self, d_model: int, num_heads: int = 1, bias: bool = True,
                 dropout: float = 0.0, layer_idx: int = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d",
                        three=3, d=self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0)
        o = rearrange(o, "b h s d -> b s (h d)")
        return self.out_proj(o)

    def state_size(self, batch_size: int = 1, sequence_length: int = 2048):
        return 2 * self.d_model * sequence_length


class ASStieltjesMHA(StieltjesMHA):
    """AS-Stieltjes (ASEntmax-recipe scaling composed with Stieltjes) as a
    causal flash mixer.

    Per-query length-adaptive scale s_i = delta + softplus(w_beta . q_i) *
    (log(i+1))^gamma (i = causal context size at position i), folded into Q
    BEFORE the kernel call — scaling q_i scales row i of QK^T, so the fused
    normalize=True Stieltjes kernel needs no changes.

    Mirrors Jack's AdaptiveScalableStieltjes (architecture_new) / ASEntmax
    (arXiv:2506.16640): beta learnable per head via query projection, gamma
    learnable, delta fixed.
    """

    def __init__(self, *args, delta: float = 1.0, gamma: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.delta = float(delta)
        self.w_beta = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self._log_gamma = nn.Parameter(
            torch.tensor(math.log(max(gamma, 1e-6))))

    def forward(self, x: torch.Tensor):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d",
                        three=3, d=self.head_dim)
        q, k, v = qkv.unbind(dim=2)                      # (B, S, H, D)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d").contiguous()
        v = rearrange(v, "b s h d -> b h s d").contiguous()

        B, H, S, D = q.shape
        # beta_i = softplus(w_beta . q_i): (B, H, S)
        beta = F.softplus(torch.einsum("bhsd,hd->bhs", q, self.w_beta))
        # causal context size at position i is (i+1)
        logn = torch.log(torch.arange(1, S + 1, device=q.device,
                                      dtype=torch.float32).clamp(min=2.0))
        scale = self.delta + beta * logn.pow(self._log_gamma.exp())  # (B,H,S)
        q = (q * scale.unsqueeze(-1).to(q.dtype)).contiguous()

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
        ).to(in_dtype)

        o = rearrange(o, "b h s d -> b s (h d)")
        o = F.dropout(o, self.dropout_p if self.training else 0.0)
        return self.out_proj(o)
