"""
nanoGPT model with swappable softmax / Stieltjes attention.

Architecture: GPT-2 style
  token embedding + position embedding → N transformer blocks → layer norm → LM head

Default config: vocab_size=258, block_size=512, n_layer=6, n_head=6, n_embd=384, dropout=0.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stieltjes attention import (requires CUDA / Triton)
# ---------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "triton"))
try:
    from stieltjes_flash_attn import stieltjes_attention as _stieltjes_attention
    from stieltjes_flash_attn import stieltjes_attention_ref as _stieltjes_attention_ref
    _STIELTJES_AVAILABLE = True
except Exception:
    _STIELTJES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 258
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    attn_type: str = "softmax"   # "softmax" or "stieltjes"
    stieltjes_q: float = 1.0
    stieltjes_num_iter: int = 3
    stieltjes_use_triton: bool = False  # False = PyTorch ref (stable for training), True = Triton kernel (fast inference)
    pos_enc: str = "learned"  # "learned" (default, GPT-2 wpe) or "none" (NoPE — needed for length-extrapolated eval)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.attn_type = config.attn_type
        self.stieltjes_q = config.stieltjes_q
        self.stieltjes_num_iter = config.stieltjes_num_iter
        self.stieltjes_use_triton = config.stieltjes_use_triton
        self.dropout_p = config.dropout

        # Single projection for Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Project to Q, K, V and split
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        sm_scale = 1.0 / math.sqrt(self.head_dim)

        if self.attn_type == "stieltjes":
            if self.stieltjes_use_triton:
                # Triton kernel — fast but backward pass has numerical issues
                y = _stieltjes_attention(
                    q, k, v,
                    causal=True,
                    sm_scale=sm_scale,
                    stieltjes_q=self.stieltjes_q,
                    num_iter=self.stieltjes_num_iter,
                )
            else:
                # PyTorch reference — stable for training via autograd
                y = _stieltjes_attention_ref(
                    q, k, v,
                    sm_scale=sm_scale,
                    causal=True,
                    stieltjes_q=self.stieltjes_q,
                    num_iter=self.stieltjes_num_iter,
                )
        else:
            # Standard softmax attention with causal mask
            # Scores: (B, H, T, T)
            att = (q @ k.transpose(-2, -1)) * sm_scale

            # Build causal mask on the fly
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            # Note: attention dropout disabled for fair comparison with Stieltjes
            # path, which computes attention internally without dropout.
            # Both paths still get resid_dropout on the output projection.

            y = att @ v  # (B, H, T, D)

        # Re-assemble heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # pre-norm attention + residual
        x = x + self.mlp(self.ln_2(x))    # pre-norm MLP + residual
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        )
        if config.pos_enc == "learned":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_enc != "none":
            raise ValueError(f"Unknown pos_enc: {config.pos_enc!r} (expected 'learned' or 'none')")
        self.transformer = nn.ModuleDict(modules)

        # LM head (no bias); weight-tied to token embedding
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) as in GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        B, T = idx.shape
        # The block_size assertion only applies when learned positional embeddings are used,
        # because the wpe table has fixed length. NoPE has no such constraint.
        if self.config.pos_enc == "learned":
            assert T <= self.config.block_size, (
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        if self.config.pos_enc == "learned":
            pos = torch.arange(T, dtype=torch.long, device=idx.device)  # (T,)
            pos_emb = self.transformer.wpe(pos)   # (T, n_embd) — broadcasts over B
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=257,  # PAD token — don't train on padding
            )
            return logits, loss

        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_model():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for attn_type in ["softmax", "stieltjes"]:
        if attn_type == "stieltjes" and not _STIELTJES_AVAILABLE:
            print(f"  [stieltjes] skipped — Triton not available on this device")
            continue
        if attn_type == "stieltjes" and not torch.cuda.is_available():
            print(f"  [stieltjes] skipped — CUDA not available")
            continue
        cfg = GPTConfig(vocab_size=258, block_size=128, n_layer=2, n_head=2, n_embd=64,
                        attn_type=attn_type, stieltjes_q=1.0)
        model = GPT(cfg).to(device)
        x = torch.randint(0, 258, (2, 64), device=device)
        logits = model(x)
        assert logits.shape == (2, 64, 258), f"Bad shape: {logits.shape}"
        print(f"  [{attn_type}] forward OK, logits shape {logits.shape}")
    # Test backward with stieltjes
    if _STIELTJES_AVAILABLE and torch.cuda.is_available():
        cfg = GPTConfig(vocab_size=258, block_size=128, n_layer=2, n_head=2, n_embd=64,
                        attn_type="stieltjes", stieltjes_q=2.0)
        model = GPT(cfg).to(device)
        x = torch.randint(0, 258, (2, 64), device=device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        print("  [stieltjes] backward OK")


if __name__ == "__main__":
    test_model()
