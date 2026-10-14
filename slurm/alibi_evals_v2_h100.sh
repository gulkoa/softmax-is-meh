#!/bin/bash
#SBATCH --job-name=alibi-evals
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/alibi-evals-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/alibi-evals-%j.err
# ALiBi 124M baseline, all length evals from the tok32k worktree (which
# carries --alibi): (1) PG-19 deep-tail ppl + bits per byte on the same
# 2.4M characters as the E/F table (job 15511190); (2) passkey 1-8x, same
# protocol as E/F (job 15511211); (3) context utilization within the
# 1024 window + window/deep-tail ppl to 8x for ALiBi, E and F.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
A=$D/ckpt_gpt2-sdpa-alibi-qk-modern-nocap-qkpilot-lr0.001_s0.pt
E=$D/ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt
F=$D/ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt
PY="uv run --project softmax-is-meh/triton --no-sync python /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts"
echo "=== (1) PG-19 ppl + BPB (identical text)"; $PY/eval_longctx_scaled.py $A --text-chars 2400000 --alpha 0.25,0.3,0.35,0.5
echo "=== (2) passkey"; HF_HUB_OFFLINE=1 $PY/eval_passkey.py $A --alpha 0,0.35 --trials 20 --show-fails 3
echo "=== (3) context utilization: ALiBi, E, F"; $PY/eval_longctx_gpt2.py $A $E $F
echo "ALL DONE"
