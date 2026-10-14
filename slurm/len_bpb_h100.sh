#!/bin/bash
#SBATCH --job-name=len-bpb
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/len-bpb-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/len-bpb-%j.err
# Tokenizer-independent length eval: PG-19 deep-tail ppl AND bits per byte on
# the SAME first 2.4M characters of the test books, for E/F (GPT-2 tokenizer)
# and E'/F' (stilt-tok-32k-v1). tok32k worktree eval.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
for c in ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt \
         ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt; do
  echo "=== $c"
  uv run --project softmax-is-meh/triton --no-sync python \
    /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/eval_longctx_scaled.py \
    $D/$c --text-chars 2400000 --alpha 0.25,0.3,0.35,0.5
done
echo "ALL DONE"
