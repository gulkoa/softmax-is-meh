#!/bin/bash
#SBATCH --job-name=passkey-pilots
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-pilots-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-pilots-%j.err
# Passkey retrieval, 1-8x context, for the 124M modern pair under both
# tokenizers (E, F, E', F'); tok32k worktree eval (trainer checkpoints).
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline HF_HUB_OFFLINE=1
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
for c in ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt \
         ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt; do
  echo "=== $c"
  uv run --project softmax-is-meh/triton --no-sync python \
    /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/eval_passkey.py $D/$c --alpha 0,0.35 --trials 20
done
echo "ALL DONE"
