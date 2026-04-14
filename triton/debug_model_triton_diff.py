"""Load the trained q=4 NoPE needle model and compare Triton vs ref outputs
on actual data, varying num_iter. Pin down why num_iter=10 fixes the random-
tensor test but the real model eval still gives 0.002.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / "nanogpt"))

from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD, SEPARATOR  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DEVICE = torch.device("cuda")
CKPT = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/"
            "results/needle_stieltjes_q4.0_seq2048_nope_ascend/checkpoint.pt")


def build_model(use_triton: bool, num_iter: int):
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


def accuracy(model, loader):
    n_corr = n_tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=-1)
            for i in range(x.shape[0]):
                sep = (x[i] == SEPARATOR).nonzero(as_tuple=True)[0]
                if len(sep) == 0:
                    continue
                out_start = sep[-1].item()
                mask = torch.zeros_like(y[i], dtype=torch.bool)
                mask[out_start:] = True
                mask &= (y[i] != PAD)
                n_corr += ((preds[i] == y[i]) & mask).sum().item()
                n_tot += mask.sum().item()
    return n_corr / n_tot if n_tot else 0.0


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    val_cfg = TaskConfig(
        task_name="needle", seq_len=2048, max_arr_len=2040,
        max_val=64, num_samples=500,
    )
    val_ds = CLRSDataset(val_cfg, seed=43)
    loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    for (use_triton, ni) in [(False, 3), (True, 3), (True, 5), (True, 10), (True, 20), (True, 50)]:
        m = build_model(use_triton, ni)
        acc = accuracy(m, loader)
        tag = "triton" if use_triton else "ref"
        print(f"{tag:>6s}  num_iter={ni:>2d}  accuracy_fixed={acc:.4f}")


if __name__ == "__main__":
    main()
