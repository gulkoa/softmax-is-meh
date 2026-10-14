#!/bin/bash
#SBATCH --job-name=passkey-diag
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-diag-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-diag-%j.err
# Diagnose the 124M in-context passkey gap (job 15511211: F 0.60/0.00/0.05 by
# depth vs E 1.00): print failed completions at 1x for F and F' (softmax) and
# E (control), to rule out a decode/format artifact before any claim.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline HF_HUB_OFFLINE=1
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
PY="uv run --project softmax-is-meh/triton --no-sync python /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts"
for c in ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt; do
  echo "=== $c"; $PY/eval_passkey.py $D/$c --lens 1024 --depths 0.1,0.3,0.5,0.7,0.9 --alpha 0 --trials 20 --show-fails 4
done
echo "ALL DONE"
