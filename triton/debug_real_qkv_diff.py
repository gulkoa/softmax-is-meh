"""Hook trained model's actual Q/K/V out of layer 0 attention, then compare
stieltjes_attention (Triton) vs stieltjes_attention_ref (PyTorch) outputs
on those real tensors. Should localize whether the bug is in the kernel or
elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / "nanogpt"))

from data import CLRSDataset, TaskConfig, VOCAB_SIZE  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref  # noqa: E402

DEVICE = torch.device("cuda")
CKPT = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/"
            "results/needle_stieltjes_q4.0_seq2048_nope_ascend/checkpoint.pt")


def build():
    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=2048,
        n_layer=6, n_head=6, n_embd=384, dropout=0.0,
        attn_type="stieltjes", stieltjes_q=4.0, stieltjes_num_iter=3,
        stieltjes_use_triton=False, pos_enc="none",
    )
    m = GPT(cfg).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def main():
    model = build()

    # Hook to capture Q/K/V tensors just before the attention op in layer 0.
    captured = {}

    attn_mod = model.transformer.h[0].attn
    orig_forward = attn_mod.forward

    def hooked_forward(x):
        # Replicate the QKV projection manually.
        B, T, C = x.size()
        qkv = attn_mod.c_attn(x)
        q, k, v = qkv.split(attn_mod.n_embd, dim=2)
        head_dim = attn_mod.n_embd // attn_mod.n_head
        q = q.view(B, T, attn_mod.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, attn_mod.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, attn_mod.n_head, head_dim).transpose(1, 2)
        captured["q"] = q.detach().clone()
        captured["k"] = k.detach().clone()
        captured["v"] = v.detach().clone()
        return orig_forward(x)

    attn_mod.forward = hooked_forward

    # Run one forward pass to populate captured Q/K/V.
    torch.manual_seed(42)
    val_ds = CLRSDataset(
        TaskConfig(task_name="needle", seq_len=2048, max_arr_len=2040,
                   max_val=64, num_samples=4),
        seed=43,
    )
    loader = DataLoader(val_ds, batch_size=4)
    x, _ = next(iter(loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        _ = model(x)

    q = captured["q"]
    k = captured["k"]
    v = captured["v"]
    print(f"captured Q/K/V shape: {tuple(q.shape)}  dtype: {q.dtype}")
    print(f"Q stats: mean={q.mean().item():.4f}  std={q.std().item():.4f}  max={q.abs().max().item():.3f}")

    sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Compare on REAL trained Q/K/V, varying num_iter.
    for ni in [3, 5, 10, 20, 50]:
        with torch.no_grad():
            out_ref = stieltjes_attention_ref(q, k, v, sm_scale=sm_scale,
                                              causal=True, stieltjes_q=4.0,
                                              num_iter=ni)
            out_tri = stieltjes_attention(q, k, v, causal=True,
                                          sm_scale=sm_scale, stieltjes_q=4.0,
                                          num_iter=ni)
        diff = (out_ref.float() - out_tri.float()).abs()
        agree = (out_ref.argmax(-1) == out_tri.argmax(-1)).float().mean().item()
        print(f"num_iter={ni:>2d}  max_abs={diff.max().item():.5f}  "
              f"mean_abs={diff.mean().item():.6f}  "
              f"ref_magnitude={out_ref.abs().max().item():.3f}  "
              f"tri_magnitude={out_tri.abs().max().item():.3f}  "
              f"argmax_agree={agree:.4f}")


if __name__ == "__main__":
    main()
