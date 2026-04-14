"""Compare ref-model vs triton-model logits layer-by-layer on one batch of
the actual trained q=4 needle checkpoint. Find where divergence is introduced.
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

DEVICE = torch.device("cuda")
CKPT = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/"
            "results/needle_stieltjes_q4.0_seq2048_nope_ascend/checkpoint.pt")


def build(use_triton, num_iter=10):
    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=2048,
        n_layer=6, n_head=6, n_embd=384, dropout=0.0,
        attn_type="stieltjes", stieltjes_q=4.0,
        stieltjes_num_iter=num_iter,
        stieltjes_use_triton=use_triton,
        pos_enc="none",
    )
    m = GPT(cfg).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def main():
    torch.manual_seed(42)
    val_ds = CLRSDataset(
        TaskConfig(task_name="needle", seq_len=2048, max_arr_len=2040,
                   max_val=64, num_samples=4),
        seed=43,
    )
    loader = DataLoader(val_ds, batch_size=4)
    x, y = next(iter(loader))
    x = x.to(DEVICE)

    m_ref = build(use_triton=False, num_iter=3)
    m_tri = build(use_triton=True,  num_iter=10)

    with torch.no_grad():
        # Capture per-layer outputs via hooks.
        ref_outs = {}
        tri_outs = {}

        def make_hook(d, tag):
            def h(module, _in, out):
                d[tag] = out.detach().float()
            return h

        for i, blk in enumerate(m_ref.transformer.h):
            blk.attn.register_forward_hook(make_hook(ref_outs, f"attn{i}"))
            blk.register_forward_hook(make_hook(ref_outs, f"block{i}"))
        for i, blk in enumerate(m_tri.transformer.h):
            blk.attn.register_forward_hook(make_hook(tri_outs, f"attn{i}"))
            blk.register_forward_hook(make_hook(tri_outs, f"block{i}"))

        logits_ref = m_ref(x)
        if isinstance(logits_ref, tuple):
            logits_ref = logits_ref[0]
        logits_tri = m_tri(x)
        if isinstance(logits_tri, tuple):
            logits_tri = logits_tri[0]

    def stats(a, b, name):
        a = a.float()
        b = b.float()
        nan_a = torch.isnan(a).sum().item()
        nan_b = torch.isnan(b).sum().item()
        inf_a = torch.isinf(a).sum().item()
        inf_b = torch.isinf(b).sum().item()
        d = (a - b).abs()
        print(f"{name:>10s}  max_abs={d.max().item():.4f}  mean_abs={d.mean().item():.5f}  "
              f"a_max={a.abs().max().item():.3f}  b_max={b.abs().max().item():.3f}  "
              f"nan={nan_a}/{nan_b} inf={inf_a}/{inf_b}")

    print("=== Per-layer ref vs triton divergence (first batch) ===")
    for i in range(6):
        stats(ref_outs[f"attn{i}"], tri_outs[f"attn{i}"], f"attn{i}")
        stats(ref_outs[f"block{i}"], tri_outs[f"block{i}"], f"block{i}")
    stats(logits_ref, logits_tri, "logits")
    # Classifier agreement on output positions
    preds_r = logits_ref.argmax(-1)
    preds_t = logits_tri.argmax(-1)
    agree = (preds_r == preds_t).float().mean().item()
    print(f"argmax agree over all positions: {agree:.4f}")


if __name__ == "__main__":
    main()
