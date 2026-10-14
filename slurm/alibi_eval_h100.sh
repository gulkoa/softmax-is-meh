#!/bin/bash
#SBATCH --job-name=alibi-eval
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/alibi-eval-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/alibi-eval-%j.err
# PG-19 deep-tail length eval of the ALiBi 124M pilot, same protocol and
# alpha grid as backbone E/F (jobs 14072702/03). Runs the eval from the
# `alibi` worktree so it imports the trainer that knows --alibi.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
CKPT=${1:-/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-alibi-qk-modern-nocap-qkpilot-lr0.001_s0.pt}
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/alibi/triton/dev_scripts/eval_longctx_scaled.py \
  "$CKPT" --alpha 0.25,0.3,0.35,0.5
echo "ALL DONE"
