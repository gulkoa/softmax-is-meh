"""Evaluate fixedcap-trained subtle-needle models at extreme context lengths.

For softmax, monkey-patches the explicit O(N^2) attention path to F.scaled_dot_product_attention
(flash backend, O(N) memory) so we can probe seq up to 131072 on a 40GB A100.

Mathematically equivalent for trained models (causal softmax with no dropout); the
difference is only that we don't materialize the (T, T) attention matrix.

Usage:
    python figures/eval_extreme_context.py \
        --model results/subtle_needle_1layer_softmax_seq128_fixedcap_ascend \
        --attn softmax --q 1.0 \
        --eval-seqs 32768 65536 131072 \
        --val-samples 200 --batch-size 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nanogpt"))
from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD, SEPARATOR  # noqa
from model import GPT, GPTConfig, CausalSelfAttention  # noqa


def patched_softmax_forward(self, x):
    """Replacement forward for CausalSelfAttention that uses SDPA flash for softmax."""
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    sm_scale = 1.0 / math.sqrt(self.head_dim)

    if self.attn_type == "stieltjes":
        # Use Triton kernel for stj at extreme context (O(N) memory)
        from stieltjes_flash_attn import stieltjes_attention as _stieltjes_attention
        y = _stieltjes_attention(
            q, k, v, causal=True, sm_scale=sm_scale,
            stieltjes_q=self.stieltjes_q, num_iter=self.stieltjes_num_iter,
        )
    else:
        # SDPA flash backend for softmax — O(N) memory
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)

    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
    return y


def load_model(model_dir, attn, q, seq_len, device):
    cfg_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "model.pt")
    saved = json.loads(open(cfg_path).read()) if os.path.isfile(cfg_path) else {}
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=seq_len,
        n_layer=saved.get("n_layer", 1),
        n_head=saved.get("n_head", 6),
        n_embd=saved.get("n_embd", 384),
        attn_type=attn,
        stieltjes_q=q,
        stieltjes_num_iter=saved.get("stieltjes_num_iter", 3),
        stieltjes_use_triton=True,  # always Triton for stj at extreme seq
        pos_enc=saved.get("pos_enc", "none"),
    )
    model = GPT(config).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Monkey-patch attention forward in every block
    for block in model.transformer.h:
        block.attn.forward = patched_softmax_forward.__get__(block.attn, CausalSelfAttention)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--attn", required=True, choices=["softmax", "stieltjes"])
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--eval-seqs", type=int, nargs="+", default=[32768, 65536, 131072])
    p.add_argument("--max-arr-len", type=int, default=120)
    p.add_argument("--val-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--needle-margin", default="subtle")
    p.add_argument("--out-suffix", default="extremectx")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seq_len in args.eval_seqs:
        out_json = os.path.join(args.model, f"accuracy_eval_seq{seq_len}_arr{args.max_arr_len}_{args.out_suffix}.json")
        if os.path.isfile(out_json):
            print(f"SKIP exists: {out_json}")
            continue

        try:
            model = load_model(args.model, args.attn, args.q, seq_len, device)
        except Exception as e:
            print(f"LOAD FAIL seq={seq_len}: {e}")
            continue

        ds = CLRSDataset(
            cfg=TaskConfig(
                task_name="needle",
                seq_len=seq_len,
                max_arr_len=args.max_arr_len,
                num_samples=args.val_samples,
                needle_margin=args.needle_margin,
            ),
            seed=args.seed,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        n_correct = n_total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                try:
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  OOM at seq={seq_len}: {e}")
                    break
                preds = logits.argmax(dim=-1)
                for i in range(x.shape[0]):
                    sep_positions = (x[i] == SEPARATOR).nonzero(as_tuple=True)[0]
                    if len(sep_positions) == 0:
                        continue
                    out_start = sep_positions[-1].item()
                    non_pad = y[i] != PAD
                    out_mask = torch.zeros_like(y[i], dtype=torch.bool)
                    out_mask[out_start:] = True
                    out_mask &= non_pad
                    n_correct += ((preds[i] == y[i]) & out_mask).sum().item()
                    n_total += out_mask.sum().item()

        acc = n_correct / n_total if n_total else float('nan')
        print(f"  seq={seq_len} accuracy_fixed={acc:.4f} (n={n_total})")
        with open(out_json, "w") as f:
            json.dump({
                "checkpoint": os.path.join(args.model, "model.pt"),
                "task": "needle",
                "needle_margin": args.needle_margin,
                "attn": args.attn, "q": args.q,
                "seq_len": seq_len, "max_arr_len": args.max_arr_len,
                "val_samples": args.val_samples,
                "accuracy_fixed": acc,
                "n_total": n_total,
                "kernel_path": "sdpa_flash" if args.attn == "softmax" else "triton",
            }, f, indent=2)

        # Free model between seq lengths
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
