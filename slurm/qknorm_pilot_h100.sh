#!/bin/bash
#SBATCH --job-name=qknorm-pilot
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/qknorm-pilot-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/qknorm-pilot-%j.err

# QK-norm x Stieltjes 2x2 pilot (plan: thesis/findings/2026-08-16-plan-
# qknorm-stieltjes.md). 124M NoPE, 70/15/15 mix, 5B tokens, lr 6e-4
# (stress point), seed 0. Runs from the qknorm worktree (branch off
# stilt11-trainer: rope + hygiene + the two new flags).
# Usage: sbatch qknorm_pilot_h100.sh <stj|sdpa> [extra trainer args]
#   A: stj  --scale-learnable                      (cap 15, current recipe)
#   B: stj  --scale-learnable --scale-cap 0 --qk-norm
#   C: sdpa                                        (control)
#   D: sdpa --qk-norm
set -euo pipefail
ARM="${1:?usage: sbatch qknorm_pilot_h100.sh <arm> [args]}"
shift
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/qknorm/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 12 --n-head 12 --n-embd 768 \
  --micro-bs 16 --grad-accum 32 --total-tokens 5e9 --warmup 2000 --lr 6e-4 \
  --data-mix "${MIX}" --nope --tag qkpilot "$@"
echo "ALL DONE"
