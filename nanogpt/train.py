"""
nanoGPT training loop for CLRS algorithmic tasks.

Logs per-epoch metrics (train_loss, val_loss, val_accuracy, epoch_time_s) to CSV.
No plotting — designed to run on a compute node.

Usage:
    python train.py --task sorting --attn softmax --out results/sorting_softmax
"""

import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from model import GPTConfig, GPT
from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=-1)
            mask = (y != PAD)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()
    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanoGPT on a CLRS task")
    parser.add_argument("--task", required=True, choices=["sorting", "binary_search", "bfs"])
    parser.add_argument("--attn", required=True, choices=["softmax", "stieltjes"])
    parser.add_argument("--q", type=float, default=1.0, help="Stieltjes q parameter")
    parser.add_argument("--out", required=True, help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-arr-len", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Output directory
    os.makedirs(args.out, exist_ok=True)

    # Save config
    config_dict = vars(args)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_cfg = TaskConfig(
        task_name=args.task,
        seq_len=args.seq_len,
        max_arr_len=args.max_arr_len,
        num_samples=args.train_samples,
    )
    val_cfg = TaskConfig(
        task_name=args.task,
        seq_len=args.seq_len,
        max_arr_len=args.max_arr_len,
        num_samples=args.val_samples,
    )

    print("Generating training data...")
    train_ds = CLRSDataset(train_cfg, seed=args.seed)
    print("Generating validation data...")
    val_ds = CLRSDataset(val_cfg, seed=args.seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    gpt_cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=args.seq_len,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        attn_type=args.attn,
        stieltjes_q=args.q,
    )
    model = GPT(gpt_cfg).to(device)
    print(f"Model parameters: {model.num_params():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # CSV logging
    csv_path = os.path.join(args.out, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "epoch_time_s"],
    )
    writer.writeheader()
    csv_file.flush()

    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        train_loss_sum = 0.0
        train_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at batch {train_batches}, skipping")
                optimizer.zero_grad()
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches if train_batches > 0 else float("nan")

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, targets=y)
                val_loss_sum += loss.item()
                val_batches += 1
        model.train()

        val_loss = val_loss_sum / val_batches if val_batches > 0 else float("nan")

        # --- Accuracy ---
        val_accuracy = compute_accuracy(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # --- Log ---
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_accuracy": f"{val_accuracy:.6f}",
            "epoch_time_s": f"{epoch_time:.2f}",
        }
        writer.writerow(row)
        csv_file.flush()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"time={epoch_time:.1f}s"
        )

    csv_file.close()

    # Save final model
    model_path = os.path.join(args.out, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
