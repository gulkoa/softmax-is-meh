#!/bin/bash
#SBATCH --job-name=stilt12-passkey
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-passkey-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-passkey-%j.err
# stilt.1.2 reading 4 (thesis/findings/2026-10-14-plan-stilt12-355m-pair.md):
# passkey retrieval at 1-8x the training context, reported without a
# threshold. The standard protocol of the 124M pilots and the 355M matched
# pair (experiments/2026-10-14-passkey-retrieval-bases.md): depths 0.1 /
# 0.5 / 0.9, 20 trials per cell, alpha 0 and 0.35; tok32k worktree eval on
# the trainer checkpoint. Versioned from passkey_pilots_h100.sh (15511211),
# one checkpoint per job. Refuses to run unless the checkpoint is FINAL.
# PREPARED 2026-10-20, NOT SUBMITTED (part of the user-gated stilt.1.2 work).
# Usage: sbatch slurm/stilt12_passkey_h100.sh <stj|sdpa>
set -euo pipefail
ARM=${1:?stj|sdpa}
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
case "$ARM" in
  stj) c=ckpt_gpt2-stj-q4-nope-qk-modern-nocap-stilt12-lr0.001-stilt-tok-32k-v1_s0.pt ;;
  sdpa) c=ckpt_gpt2-sdpa-nope-qk-modern-nocap-stilt12-lr0.001-stilt-tok-32k-v1_s0.pt ;;
  *) echo "arm must be stj or sdpa"; exit 2 ;;
esac
/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python - "$D/$c" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
a = ck["args"]
last = int(a["total_tokens"] // (a["micro_bs"] * a["grad_accum"] * a["ctx"])) - 1
print(sys.argv[1].split("/")[-1], "step", ck["step"], "final step", last, flush=True)
sys.exit(0 if ck["step"] >= last else 1)
PY
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline HF_HUB_OFFLINE=1
echo "=== $c"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/eval_passkey.py \
  "$D/$c" --alpha 0,0.35 --trials 20
echo "ALL DONE"
