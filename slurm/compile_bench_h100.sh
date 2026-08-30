#!/bin/bash
#SBATCH --job-name=compile-bench
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/compile-bench-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/compile-bench-%j.err
# Item-7 speed lever bench: --compile on vs off, 124M modern stack,
# ~114 steps each. Runs from the backbone worktree.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7,math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15,code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"
BASE="/users/PAS2402/alexg/softmax/.worktrees/backbone/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn stj --n-layer 12 --n-head 12 --n-embd 768 \
  --micro-bs 16 --grad-accum 32 --total-tokens 6e7 --warmup 20 --lr 6e-4 \
  --data-mix ${MIX} --nope --qk-norm --scale-cap 0 --scale-learnable --modern \
  --val-every 50 --ckpt-every 100000"
echo "=== EAGER ==="
uv run --project softmax-is-meh/triton --no-sync python $BASE --tag cbench-eager
echo "=== COMPILED ==="
uv run --project softmax-is-meh/triton --no-sync python $BASE --tag cbench-comp --compile
echo "ALL DONE"
