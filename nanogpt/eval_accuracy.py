"""Evaluate a saved checkpoint against both the fixed (output-only) accuracy
metric and the old (all-positions) metric, for direct comparison.

Usage:
    python eval_accuracy.py --checkpoint results/<run>/model.pt \
        --task binary_search --attn stieltjes --q 8.0 \
        --seq-len 128 --val-samples 5000 --seed 42 \
        [--out results/<run>/accuracy_fixed.json]

Writes one JSON object with:
    {"checkpoint": ..., "task": ..., "attn": ..., "q": ...,
     "val_samples": N, "seed": S,
     "accuracy_fixed": 0.xxx,  # output positions only (current train.py metric)
     "accuracy_all":   0.xxx,  # all non-PAD positions (inflated "old" metric)
     "accuracy_input_echo": 0.xxx,  # input positions only (should be high if model echoes)
     "output_start_positions": [n, min, max, mean]}  # sanity check

If --out is omitted, writes to `<checkpoint dir>/accuracy_fixed.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD, SEPARATOR  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def compute_all_metrics(model, loader, device, first_n_output=None):
    """Compute three variants of accuracy in one pass.

    - fixed: positions from last SEPARATOR onward (train.py current metric)
    - all: all non-PAD positions (broken old metric that inflated scores)
    - input_echo: positions before the last SEPARATOR (diagnostic)
    """
    model.eval()
    n_out_correct = n_out_total = 0
    n_all_correct = n_all_total = 0
    n_inp_correct = n_inp_total = 0
    out_starts = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=-1)

            for i in range(x.shape[0]):
                sep_positions = (x[i] == SEPARATOR).nonzero(as_tuple=True)[0]
                non_pad = y[i] != PAD
                match = preds[i] == y[i]

                # all-positions metric (old)
                n_all_correct += (match & non_pad).sum().item()
                n_all_total += non_pad.sum().item()

                if len(sep_positions) == 0:
                    continue

                out_start = sep_positions[-1].item()
                out_starts.append(out_start)

                out_mask = torch.zeros_like(y[i], dtype=torch.bool)
                if first_n_output is not None:
                    # Score only the first N output positions (e.g., skip max_idx
                    # when NoPE makes it unlearnable)
                    end_pos = min(out_start + first_n_output, y[i].shape[0])
                    out_mask[out_start:end_pos] = True
                else:
                    out_mask[out_start:] = True
                out_mask &= non_pad
                n_out_correct += (match & out_mask).sum().item()
                n_out_total += out_mask.sum().item()

                inp_mask = torch.zeros_like(y[i], dtype=torch.bool)
                inp_mask[:out_start] = True
                inp_mask &= non_pad
                n_inp_correct += (match & inp_mask).sum().item()
                n_inp_total += inp_mask.sum().item()

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    out_starts_t = torch.tensor(out_starts, dtype=torch.float) if out_starts else torch.tensor([0.0])
    return {
        "accuracy_fixed": safe_div(n_out_correct, n_out_total),
        "accuracy_all": safe_div(n_all_correct, n_all_total),
        "accuracy_input_echo": safe_div(n_inp_correct, n_inp_total),
        "output_token_count": n_out_total,
        "input_token_count": n_inp_total,
        "all_token_count": n_all_total,
        "output_start_stats": {
            "n": len(out_starts),
            "min": int(out_starts_t.min().item()),
            "max": int(out_starts_t.max().item()),
            "mean": float(out_starts_t.mean().item()),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to model.pt (state_dict)")
    p.add_argument("--task", required=True,
                   choices=["sorting", "binary_search", "bfs", "max", "needle", "top2_needle", "top3_needle"])
    p.add_argument("--attn", required=True,
                   choices=["softmax", "stieltjes", "stieltjes_jt",
                            "stieltjes_jt_init", "stieltjes_no_safeguard",
                            "stieltjes_max_eps"])
    p.add_argument("--q", type=float, default=1.0)
    # None defaults mean "take value from training config.json".
    # Pass an explicit value on the CLI to override (e.g. for length-extrapolated eval).
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--max-arr-len", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--val-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42,
                   help="Val-set seed (use 43 if reproducing train-time eval: train.py uses seed+1).")
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Eval dropout (default 0 for determinism; train used 0.1).")
    p.add_argument("--out", default=None,
                   help="Output JSON path. Default: <checkpoint_dir>/accuracy_fixed.json")
    p.add_argument("--eval-seq-len", type=int, default=None,
                   help="Evaluate at a different seq_len than training. Requires "
                        "the trained model to use NoPE (--pos-enc none) — learned "
                        "positional embeddings cannot extrapolate past block_size. "
                        "Default: use the saved config's seq_len.")
    p.add_argument("--use-triton", action="store_true",
                   help="Use the Triton stieltjes kernel for forward (faster at long "
                        "context, fwd-only). Default: PyTorch reference.")
    p.add_argument("--num-iter-override", type=int, default=None,
                   help="Override stieltjes_num_iter (default: 10 for triton, 3 for ref).")
    p.add_argument("--first-n-output", type=int, default=None,
                   help="Score only first N output tokens (e.g., 1 to skip max_idx when NoPE).")
    p.add_argument("--needle-margin", default=None, choices=["distinctive", "subtle"],
                   help="Override needle distinguishability mode for eval. Default: "
                        "take from training config.json.")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(args.out) if args.out else ckpt_path.parent / "accuracy_fixed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fill in any CLI args that weren't explicitly passed from the training
    # config.json. Explicit CLI values always win (e.g. passing
    # --max-arr-len 8184 to eval at longer context than training).
    cfg_path = ckpt_path.parent / "config.json"
    saved_pos_enc = "learned"
    saved_train_seq_len = None
    if cfg_path.is_file():
        try:
            saved = json.loads(cfg_path.read_text())
            for key in ("seq_len", "max_arr_len", "max_val", "needle_margin",
                        "n_layer", "n_head", "n_embd"):
                if getattr(args, key, None) is None and key in saved:
                    setattr(args, key, saved[key])
            saved_train_seq_len = saved.get("seq_len")
            saved_pos_enc = saved.get("pos_enc", "learned")
        except Exception as e:
            print(f"WARN: could not parse {cfg_path}: {e}", file=sys.stderr)

    # Any remaining None values fall back to legacy defaults for safety.
    if args.seq_len is None:
        args.seq_len = 128
    if args.max_arr_len is None:
        args.max_arr_len = 16
    if args.max_val is None:
        args.max_val = 64
    if getattr(args, "needle_margin", None) is None:
        args.needle_margin = "distinctive"
    if args.n_layer is None:
        args.n_layer = 6
    if args.n_head is None:
        args.n_head = 6
    if args.n_embd is None:
        args.n_embd = 384

    # Determine actual eval seq_len: --eval-seq-len overrides saved value.
    eval_seq_len = args.eval_seq_len if args.eval_seq_len is not None else args.seq_len
    if args.eval_seq_len is not None and saved_pos_enc == "learned":
        # block_size is fixed by the wpe table — extrapolation is impossible.
        if args.eval_seq_len > (saved_train_seq_len or args.seq_len):
            raise SystemExit(
                f"Cannot evaluate at seq_len={args.eval_seq_len} on a model trained "
                f"with learned positional embeddings (block_size={saved_train_seq_len}). "
                f"Retrain with --pos-enc none for length-extrapolated eval."
            )
    args.seq_len = eval_seq_len  # use eval-time seq_len for both data and block_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    val_cfg = TaskConfig(
        task_name=args.task,
        seq_len=args.seq_len,
        max_arr_len=args.max_arr_len,
        max_val=args.max_val,
        num_samples=args.val_samples,
        needle_margin=args.needle_margin,
    )
    val_ds = CLRSDataset(val_cfg, seed=args.seed + 1)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    # The Triton kernel uses a scalar LAMBDA_INIT = N^{1/q} for all rows;
    # the PyTorch ref uses per-row (i+1)^{1/q} under causal masking. At the
    # default num_iter=3 this divergence is large enough at q=4 causal to
    # destroy downstream accuracy (confirmed: ref=0.916 vs triton=0.002 on
    # the same q=4 needle checkpoint). num_iter=10 converges; use that
    # whenever we go through the Triton path.
    num_iter = args.num_iter_override if args.num_iter_override is not None else (10 if args.use_triton else 3)
    gpt_cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        attn_type=args.attn,
        stieltjes_q=args.q,
        stieltjes_num_iter=num_iter,
        pos_enc=saved_pos_enc,
        stieltjes_use_triton=args.use_triton,
    )
    model = GPT(gpt_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state and "optimizer" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    metrics = compute_all_metrics(model, val_loader, device,
                                  first_n_output=args.first_n_output)
    result = {
        "checkpoint": str(ckpt_path),
        "task": args.task,
        "attn": args.attn,
        "q": args.q,
        "seq_len": args.seq_len,
        "val_samples": args.val_samples,
        "seed": args.seed,
        **metrics,
    }

    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
