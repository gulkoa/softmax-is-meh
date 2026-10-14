#!/bin/bash
#SBATCH --job-name=alibi-pilot
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/alibi-pilot-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/alibi-pilot-%j.err
# ALiBi length baseline (advisor, 08-25; roadmap "[USER] ALiBi length
# baseline"): softmax + ALiBi on the modern recipe at 124M — byte-identical
# to backbone arm F (sdpa + qk-norm + modern, no cap, lr 1e-3, 5B tokens of
# the 70/15/15 mix) except --alibi in place of --nope. Runs the trainer
# from the `alibi` worktree (main's trainer is frozen under the queued s1
# chains). Self-resuming; a FINAL-guarded second chunk no-ops.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/alibi/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn sdpa --n-layer 12 --n-head 12 --n-embd 768 \
  --micro-bs 16 --grad-accum 32 --total-tokens 5e9 --warmup 2000 --lr 1e-3 \
  --data-mix "${MIX}" --alibi --qk-norm --scale-cap 0 --scale-learnable \
  --modern --tag qkpilot "$@"
echo "ALL DONE"
