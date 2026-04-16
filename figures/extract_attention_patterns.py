"""Extract attention weights from trained models for Velickovic-style grid plot.

For each (model, eval_seq) pair:
  - Load checkpoint
  - Build causal-mask attention hook (reuses analyze.py logic)
  - Run K samples of subtle-needle-at-eval_seq through the model
  - For each sample, capture the attention pattern from the LAST query position
    over all key positions
  - Save a (num_samples, num_keys) tensor to disk

Usage:
    python extract_attention_patterns.py \
        --model results/subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend \
        --attn stieltjes --q 8.0 \
        --eval-seqs 128 512 2048 8192 \
        --num-samples 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanogpt"))
from model import GPT, GPTConfig  # noqa
from data import CLRSDataset, TaskConfig, VOCAB_SIZE  # noqa
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanogpt"))
from analyze import _compute_stieltjes_weights  # noqa


def load_model(model_dir, attn, q, seq_len, device):
    cfg_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "model.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    saved = {}
    if os.path.isfile(cfg_path):
        saved = json.loads(open(cfg_path).read())
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=seq_len,
        n_layer=saved.get("n_layer", 6),
        n_head=saved.get("n_head", 6),
        n_embd=saved.get("n_embd", 384),
        attn_type=attn,
        stieltjes_q=q,
        pos_enc=saved.get("pos_enc", "none"),
    )
    model = GPT(config).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def capture_attention_last_query(model, x, attn, q):
    """Return (L, B, H, T) tensor of attention weights from last query over all
    key positions, captured per-layer during forward."""
    storage = []  # per-layer: appended weights

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # inputs[0] is the pre-norm input (B, T, C)
            xin = inputs[0]
            B, T, C = xin.shape
            nh = module.n_head
            hd = module.head_dim
            sm = 1.0 / math.sqrt(hd)
            with torch.no_grad():
                qkv = module.c_attn(xin)
                qh, kh, _ = qkv.split(module.n_embd, dim=2)
                qh = qh.view(B, T, nh, hd).transpose(1, 2)  # (B,H,T,D)
                kh = kh.view(B, T, nh, hd).transpose(1, 2)
                scores = (qh @ kh.transpose(-2, -1)) * sm  # (B,H,T,T)
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                             device=xin.device))
                if attn == "softmax":
                    s = scores.masked_fill(~mask, float("-inf"))
                    w = torch.softmax(s, dim=-1).nan_to_num(nan=0.0)
                else:
                    w = _compute_stieltjes_weights(scores, mask, q)
            # keep only the last query row for each sample
            storage.append(w[:, :, -1, :].cpu())  # (B, H, T)

        return hook

    handles = []
    for li, block in enumerate(model.transformer.h):
        handles.append(block.attn.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    return torch.stack(storage, dim=0)  # (L, B, H, T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model dir containing model.pt + config.json")
    p.add_argument("--attn", required=True, choices=["softmax", "stieltjes"])
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--eval-seqs", type=int, nargs="+", default=[128, 512, 2048, 8192])
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--needle-margin", default="subtle")
    p.add_argument("--max-val", type=int, default=64)
    p.add_argument("--out-subdir", default="attn_patterns")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(args.model, args.out_subdir)
    os.makedirs(out_root, exist_ok=True)

    for seq_len in args.eval_seqs:
        arr_len = seq_len - 8
        out_pt = os.path.join(out_root, f"attn_last_query_seq{seq_len}.pt")
        if os.path.isfile(out_pt):
            print(f"SKIP exists: {out_pt}")
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
                max_arr_len=arr_len,
                max_val=args.max_val,
                num_samples=args.num_samples,
                needle_margin=args.needle_margin,
            ),
            seed=43,
        )
        # Build batch — __getitem__ returns (x, y)
        xs = []
        for i in range(args.num_samples):
            x_i, _ = ds[i]
            xs.append(x_i)
        x = torch.stack(xs, dim=0).to(device)

        print(f"Capturing attn for seq={seq_len} x.shape={tuple(x.shape)} ...")
        try:
            attn = capture_attention_last_query(model, x, args.attn, args.q)
            # attn: (L, B, H, T)
            torch.save(
                {
                    "attn_last_query": attn,
                    "eval_seq": seq_len,
                    "attn_type": args.attn,
                    "q": args.q,
                    "num_samples": args.num_samples,
                },
                out_pt,
            )
            print(f"  saved {out_pt}  shape={tuple(attn.shape)}")
        except Exception as e:
            print(f"  CAPTURE FAIL: {e}")


if __name__ == "__main__":
    main()
