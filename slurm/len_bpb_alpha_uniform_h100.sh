#!/bin/bash
#SBATCH --job-name=len-bpb-uni
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/len-bpb-alpha-uniform-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/len-bpb-alpha-uniform-%j.err
# Uniform 0.025 alpha grid, 0.05-0.70, for the tokenizer-independent length
# eval of E/F (GPT-2 tokenizer) and E'/F' (stilt-tok-32k-v1): deep-tail ppl and
# bits per byte on the same first 2.4M characters of the PG-19 test books
# (record thesis/experiments/2026-10-18-len-bpb-alpha-uniform-124m.md).
# Versioned copy of len_bpb_h100.sh (job 15511190); same checkpoints, order
# and tok32k-worktree eval script (unchanged since 2f9b546, before 15511190);
# only the alpha grid, partition, time and log names differ. Its 0.25, 0.3,
# 0.35 and 0.5 columns must reproduce 15511190.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
for c in ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt \
         ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt; do
  echo "=== $c"
  uv run --project softmax-is-meh/triton --no-sync python \
    /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/eval_longctx_scaled.py \
    $D/$c --text-chars 2400000 --alpha 0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7
done
echo "ALL DONE"
