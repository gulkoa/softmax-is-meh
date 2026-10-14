#!/bin/bash
#SBATCH --job-name=tok32k-eval
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/tok32k-eval-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/tok32k-eval-%j.err
# Tokenizer pilot evaluation (tok32k worktree): bits per byte on the same
# 1,500 validation documents per source for E, F (GPT-2 tokenizer) and
# E', F' (stilt-tok-32k-v1); then the tokenizer-aware PG-19 length eval of
# E' and F' with the E/F alpha grid.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
# (no HF offline flag: the PG-19 stream loader reaches the Hub like the earlier length evals did)
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
PY="uv run --project softmax-is-meh/triton --no-sync python /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts"
if [ -z "${SKIP_BPB:-}" ]; then
for c in ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt \
         ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt; do
  echo "=== BPB $c"; $PY/eval_bpb.py $D/$c --docs 1500
done
fi
for c in ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt; do
  echo "=== LENGTH $c"; $PY/eval_longctx_scaled.py $D/$c --alpha 0.25,0.3,0.35,0.5
done
echo "ALL DONE"
