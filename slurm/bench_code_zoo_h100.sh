#!/bin/bash
#SBATCH --job-name=bench-code-zoo
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:45:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-code-zoo-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-code-zoo-%j.err

# Family-wide code benchmark sweep (2026-08-11, user directive:
# "best model for coding / benchmark everything / checkpoints
# everywhere"). Loops the PROVEN eval_code_stilt.py invocation over a
# list of ckpts. Usage:
#   sbatch bench_code_zoo_h100.sh both  ckptA ckptB ...   (MBPP+HumanEval, limit 100)
#   sbatch bench_code_zoo_h100.sh humaneval ckptA ...     (HumanEval only)
# Skips a ckpt if its humaneval JSON already exists (idempotent resubmit).
# Data-only job: JSONs land next to each ckpt; wandb push + ranking run
# on the login node afterwards.

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

# Sweep isolation: pre-sweep codeeval JSONs (older RL harvests, different
# protocol: full MBPP test) are backed up in ${SWEEP}/pre_sweep_backup/.
# eval_code_stilt.py writes next to the ckpt (fixed path), so each fresh
# JSON is copied into ${SWEEP}/ and completion is marked by a sentinel
# there — never by ckpt-adjacent files, which may predate the sweep.
SWEEP=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_zoo_20260811

DATASET="${1:?usage: bench_code_zoo_h100.sh <both|humaneval|mbpp> ckpt...}"
shift

for CKPT in "$@"; do
  BASE=$(basename "${CKPT}" .pt)
  SENTINEL="${SWEEP}/${BASE}.${DATASET}.done"
  if [[ -f "${SENTINEL}" ]]; then
    echo "SKIP (sweep done): ${CKPT}"
    continue
  fi
  echo "=== EVAL ${DATASET} :: ${CKPT} ==="
  if uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/eval_code_stilt.py "${CKPT}" \
    --dataset "${DATASET}" --n 20 --k 1,10 --limit 100; then
    # copy only the JSONs this run refreshed (an old ckpt-adjacent JSON
    # for the OTHER dataset would smuggle mixed protocols into the sweep)
    if [[ "${DATASET}" == "both" || "${DATASET}" == "humaneval" ]]; then
      cp "${CKPT%.pt}_codeeval_humaneval.json" "${SWEEP}/"
    fi
    if [[ "${DATASET}" == "both" || "${DATASET}" == "mbpp" ]]; then
      cp "${CKPT%.pt}_codeeval_mbpp.json" "${SWEEP}/"
    fi
    touch "${SENTINEL}"
  else
    echo "EVAL FAILED (continuing): ${CKPT}"
  fi
done
echo "ALL DONE"
