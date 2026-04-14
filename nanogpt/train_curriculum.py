"""Train nanoGPT with a q-curriculum for Stieltjes attention.

The Stieltjes q parameter is mutated epoch-by-epoch according to a schedule.
Motivating hypothesis: high-q Stieltjes is hard attention, which has vanishing
gradients at random init. A curriculum from low q (softmax-like) to the target
high q lets the model first learn a reasonable Q/K representation, then sharpen.

Usage:
    python train_curriculum.py \
        --task binary_search --attn stieltjes \
        --q-schedule "1@1,2@11,4@21,8@31,16@41" \
        --out results/binary_search_curriculum_q1to16_ascend/ \
        --epochs 50

Schedule format: comma-separated `q@start_epoch`, 1-indexed. Each entry sets
the q value starting at its epoch, held until the next entry. Example above:
q=1 for epochs 1-10, q=2 for 11-20, q=4 for 21-30, q=8 for 31-40, q=16 for
41-50.

Metrics CSV adds a `q` column so the trajectory is reconstructible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD, SEPARATOR  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from train import compute_accuracy  # noqa: E402  (reuse fixed metric)


def parse_schedule(s: str) -> list[tuple[int, float]]:
    """Parse 'q1@e1,q2@e2,...' into a sorted list of (start_epoch, q)."""
    entries = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        q_str, epoch_str = part.split("@")
        entries.append((int(epoch_str), float(q_str)))
    entries.sort()
    if not entries:
        raise ValueError("Empty schedule")
    if entries[0][0] != 1:
        raise ValueError(f"Schedule must start at epoch 1, got {entries[0][0]}")
    return entries


def parse_schedule_file(path: str) -> list[tuple[int, float]]:
    """Parse a file with 'epoch:q' per line (comments with # OK)."""
    entries = []
    for ln, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        epoch_str, q_str = line.split(":")
        entries.append((int(epoch_str), float(q_str)))
    entries.sort()
    if not entries or entries[0][0] != 1:
        raise ValueError(f"Schedule file {path} must start at epoch 1")
    return entries


def q_at_epoch(schedule: list[tuple[int, float]], epoch: int) -> float:
    """Return the q value active at `epoch`."""
    active_q = schedule[0][1]
    for start, q in schedule:
        if start <= epoch:
            active_q = q
        else:
            break
    return active_q


def set_q(model: torch.nn.Module, q: float) -> None:
    """Mutate stieltjes_q on every attention block."""
    for m in model.modules():
        if hasattr(m, "stieltjes_q"):
            m.stieltjes_q = q


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["sorting", "binary_search", "bfs", "max", "needle"])
    p.add_argument("--attn", default="stieltjes", choices=["softmax", "stieltjes"],
                   help="Curriculum is meaningful only for stieltjes.")
    p.add_argument("--q-schedule", default=None,
                   help="Inline schedule 'q1@e1,q2@e2,...' (1-indexed).")
    p.add_argument("--q-schedule-file", default=None,
                   help="Path to schedule file, 'epoch:q' per line.")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--max-arr-len", type=int, default=16)
    p.add_argument("--max-val", type=int, default=64)
    p.add_argument("--train-samples", type=int, default=50000)
    p.add_argument("--val-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-each-block", action="store_true",
                   help="Run eval_accuracy JSON dump at every schedule boundary.")
    args = p.parse_args()
    if (args.q_schedule is None) == (args.q_schedule_file is None):
        p.error("Provide exactly one of --q-schedule / --q-schedule-file")
    return args


def main():
    args = parse_args()

    if args.q_schedule is not None:
        schedule = parse_schedule(args.q_schedule)
    else:
        schedule = parse_schedule_file(args.q_schedule_file)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    # Save config (includes the schedule)
    config_dict = {**vars(args), "schedule": [[e, q] for e, q in schedule]}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Schedule: {schedule}")

    train_cfg = TaskConfig(task_name=args.task, seq_len=args.seq_len,
                           max_arr_len=args.max_arr_len, max_val=args.max_val,
                           num_samples=args.train_samples)
    val_cfg = TaskConfig(task_name=args.task, seq_len=args.seq_len,
                         max_arr_len=args.max_arr_len, max_val=args.max_val,
                         num_samples=args.val_samples)

    print("Generating training data...")
    train_ds = CLRSDataset(train_cfg, seed=args.seed)
    print("Generating validation data...")
    val_ds = CLRSDataset(val_cfg, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    # Initialize model at the first q value
    initial_q = schedule[0][1]
    gpt_cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=args.seq_len,
                        n_layer=6, n_head=6, n_embd=384, dropout=0.1,
                        attn_type=args.attn, stieltjes_q=initial_q)
    model = GPT(gpt_cfg).to(device)
    print(f"Model parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    csv_path = os.path.join(args.out, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "q", "train_loss", "val_loss", "val_accuracy", "epoch_time_s"],
    )
    writer.writeheader()
    csv_file.flush()

    schedule_boundaries = {e for e, _ in schedule}
    current_q = None

    model.train()
    for epoch in range(1, args.epochs + 1):
        # Update q if this epoch is a schedule boundary
        q_this_epoch = q_at_epoch(schedule, epoch)
        if q_this_epoch != current_q:
            set_q(model, q_this_epoch)
            current_q = q_this_epoch
            print(f"[schedule] epoch {epoch}: q -> {current_q}")

        epoch_start = time.time()

        train_loss_sum = 0.0
        train_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARN: NaN/Inf loss at batch {train_batches}, skip")
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches if train_batches else float("nan")

        # val loss
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
        val_loss = val_loss_sum / val_batches if val_batches else float("nan")

        val_accuracy = compute_accuracy(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        writer.writerow({
            "epoch": epoch,
            "q": f"{current_q:.4f}",
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_accuracy": f"{val_accuracy:.6f}",
            "epoch_time_s": f"{epoch_time:.2f}",
        })
        csv_file.flush()

        print(f"Epoch {epoch:3d}/{args.epochs} | q={current_q:.3g} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_accuracy:.4f} | time={epoch_time:.1f}s")

        # Per-boundary checkpoint so we have a model at each q step
        if args.eval_each_block and (epoch in schedule_boundaries or epoch == args.epochs):
            ckpt = os.path.join(args.out, f"model_epoch{epoch:03d}_q{current_q:g}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  [checkpoint] {ckpt}")

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out, "checkpoint.pt")
            tmp = ckpt_path + ".tmp"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "config": config_dict,
                        "current_q": current_q}, tmp)
            os.replace(tmp, ckpt_path)

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    print(f"Done. Artifacts in {args.out}")


if __name__ == "__main__":
    main()
