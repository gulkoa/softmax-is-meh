"""Quick test: does Stieltjes attention training diverge with the PyTorch
reference implementation (not the Triton kernel)?

If it diverges with PyTorch autograd too, the problem is mathematical
(gradient dynamics of algebraic normalization), not a kernel bug.
If it only diverges with the Triton kernel, the problem is numerical
precision in the kernel backward pass.
"""

import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "triton"))
from stieltjes_flash_attn import stieltjes_attention_ref

from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD


class SimpleAttnBlock(nn.Module):
    """Single attention block using the PyTorch reference Stieltjes implementation."""
    def __init__(self, n_embd, n_head, sq=1.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.sq = sq

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln(x)
        qkv = self.c_attn(h)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        sm_scale = 1.0 / math.sqrt(self.head_dim)
        # Use PyTorch reference — autograd handles backward
        o = stieltjes_attention_ref(q, k, v, sm_scale, causal=True,
                                     stieltjes_q=self.sq, num_iter=5, eps=1e-6)
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return x + self.c_proj(o)


class TinyModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2, sq=1.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(128, n_embd)
        self.blocks = nn.ModuleList([SimpleAttnBlock(n_embd, n_head, sq) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def test_training_stability(device="cuda", sq=1.0, lr=3e-4, epochs=15):
    print(f"\n{'='*60}")
    print(f"Testing PyTorch-ref Stieltjes training: q={sq}, lr={lr}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    cfg = TaskConfig(task_name="sorting", seq_len=64, num_samples=500)
    ds = CLRSDataset(cfg, seed=42)

    model = TinyModel(VOCAB_SIZE, n_embd=128, n_head=4, n_layer=2, sq=sq).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

            if torch.isnan(loss):
                print(f"  Epoch {epoch}: NaN loss! Diverged.")
                return False

            optimizer.zero_grad()
            loss.backward()

            # Check gradient norms
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / n
        print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}  grad_norm={grad_norm:.4f}")

    print(f"  Survived {epochs} epochs without NaN!")
    return True


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test PyTorch reference at same lr that breaks the Triton kernel
    test_training_stability(device, sq=1.0, lr=3e-4, epochs=15)
    test_training_stability(device, sq=1.0, lr=1e-4, epochs=15)
    test_training_stability(device, sq=2.0, lr=3e-4, epochs=15)
    test_training_stability(device, sq=4.0, lr=3e-4, epochs=15)
