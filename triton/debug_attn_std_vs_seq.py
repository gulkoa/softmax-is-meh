"""Log per-layer attention-output std across eval_seq_len for softmax and
stj q=4 NoPE checkpoints. Visualizes the distribution-shift mechanism
behind the length-extrapolation failure in stj: if stj's output std grows
with eval_seq while softmax's stays stable, that's the smoking gun.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / "nanogpt"))

from data import CLRSDataset, TaskConfig, VOCAB_SIZE  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DEVICE = torch.device("cuda")

CONFIGS = [
    ("softmax", 1.0, "results/needle_softmax_q1.0_seq2048_nope_ascend/checkpoint.pt"),
    ("stieltjes", 4.0, "results/needle_stieltjes_q4.0_seq2048_nope_ascend/checkpoint.pt"),
]
EVAL_SEQS = [2048, 4096, 8192, 16384]


def build(attn, q):
    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=16384,
        n_layer=6, n_head=6, n_embd=384, dropout=0.0,
        attn_type=attn, stieltjes_q=q, stieltjes_num_iter=3,
        stieltjes_use_triton=False, pos_enc="none",
    )
    return GPT(cfg).to(DEVICE)


def main():
    for attn, q, ckpt_path in CONFIGS:
        ckpt_path = Path("/users/PAS2402/alexg/softmax/softmax-is-meh") / ckpt_path
        if not ckpt_path.is_file():
            print(f"SKIP {attn} q={q}: no checkpoint at {ckpt_path}")
            continue
        state = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        for eval_seq in EVAL_SEQS:
            torch.manual_seed(42)
            # Attention matrix scaling: bs=1 for long seqs to avoid OOM.
            bs = 4 if eval_seq <= 2048 else 2 if eval_seq <= 4096 else 1
            val_ds = CLRSDataset(
                TaskConfig(task_name="needle", seq_len=eval_seq,
                           max_arr_len=eval_seq - 8, max_val=64,
                           num_samples=bs * 4),
                seed=43,
            )
            loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

            model = build(attn, q)
            model.load_state_dict(state)
            model.eval()

            # Hooks on per-layer attention output.
            captured = {}

            def make_hook(i):
                def h(_m, _in, out):
                    captured[i] = out.detach().float()
                return h

            handles = []
            for i, blk in enumerate(model.transformer.h):
                handles.append(blk.attn.register_forward_hook(make_hook(i)))

            try:
                with torch.no_grad():
                    for x, _ in loader:
                        model(x.to(DEVICE))
                        break
            except torch.cuda.OutOfMemoryError:
                print(f"OOM {attn} q={q} eval_seq={eval_seq}")
                for h in handles:
                    h.remove()
                del model
                torch.cuda.empty_cache()
                continue

            stds = [captured[i].std().item() for i in range(6)]
            mags = [captured[i].abs().max().item() for i in range(6)]
            print(f"{attn:>9s} q={q:>3.1f} eval_seq={eval_seq:>5d}  "
                  f"per-layer std: " + ", ".join(f"{s:.3f}" for s in stds) +
                  f"  | max_abs: " + ", ".join(f"{m:.2f}" for m in mags))

            for h in handles:
                h.remove()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
