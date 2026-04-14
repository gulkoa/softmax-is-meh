"""Does REF accuracy also degrade when num_iter changes? If model is overfit
to ref's specific partially-converged num_iter=3 output, then ref at higher
num_iter should also give worse accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / "nanogpt"))

from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD, SEPARATOR  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DEVICE = torch.device("cuda")
CKPT = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/"
            "results/needle_stieltjes_q4.0_seq2048_nope_ascend/checkpoint.pt")


def build(use_triton, num_iter):
    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=2048,
        n_layer=6, n_head=6, n_embd=384, dropout=0.0,
        attn_type="stieltjes", stieltjes_q=4.0, stieltjes_num_iter=num_iter,
        stieltjes_use_triton=use_triton, pos_enc="none",
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
    val_ds = CLRSDataset(
        TaskConfig(task_name="needle", seq_len=2048, max_arr_len=2040,
                   max_val=64, num_samples=500),
        seed=43,
    )
    loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    for use_triton in [False, True]:
        for ni in [3, 5, 10, 20, 50]:
            m = build(use_triton, ni)
            acc = accuracy(m, loader)
            tag = "triton" if use_triton else "ref   "
            print(f"{tag}  num_iter={ni:>2d}  accuracy_fixed={acc:.4f}")


if __name__ == "__main__":
    main()
