#!/bin/bash
#SBATCH --job-name=stj-alibi
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stj-alibi-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stj-alibi-%j.err
# Stieltjes + ALiBi 124M pilot (pre-registration
# thesis/experiments/2026-10-14-stj-alibi-124m.md): the softmax + ALiBi recipe
# (alibi_pilot_h100.sh) with --attn stj --stj-iters 5. Runs the stjalibi
# worktree trainer, which imports the stjalibi worktree's kernel (printed at
# startup). Self-resuming, FINAL-guarded.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/stjalibi/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn stj --stj-iters 5 --n-layer 12 --n-head 12 --n-embd 768 \
  --micro-bs 16 --grad-accum 32 --total-tokens 5e9 --warmup 2000 --lr 1e-3 \
  --data-mix "${MIX}" --alibi --qk-norm --scale-cap 0 --scale-learnable \
  --modern --tag qkpilot "$@"
echo "ALL DONE"
