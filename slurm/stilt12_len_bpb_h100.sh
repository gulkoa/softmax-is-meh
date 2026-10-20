#!/bin/bash
#SBATCH --job-name=stilt12-len
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-len-bpb-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-len-bpb-%j.err
# stilt.1.2 reading 3 (thesis/findings/2026-10-14-plan-stilt12-355m-pair.md, protocol
# note of 2026-10-17 and scoring note of 2026-10-20): deep-tail bits per byte
# on the same first 2.4M characters of the PG-19 test books, uniform alpha
# grid every 0.025 from 0.05 to 0.7. Scored on 0.05-0.5; the 0.525-0.7
# columns are consulted only for an arm whose optimum sits at 0.5
# (scripts/score_stilt12.py --length <logs>). Versioned from
# len_bpb_alpha_uniform_h100.sh (job 15638836): same tok32k-worktree eval
# script and grid, one stilt.1.2 checkpoint per job so each fits the debug
# limit. Refuses to run unless the checkpoint is FINAL.
# PREPARED 2026-10-20, NOT SUBMITTED (part of the user-gated stilt.1.2 work).
# Usage: sbatch slurm/stilt12_len_bpb_h100.sh <stj|sdpa>
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
export WANDB_MODE=offline
echo "=== $c"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/eval_longctx_scaled.py \
  "$D/$c" --text-chars 2400000 --alpha 0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7
echo "ALL DONE"
