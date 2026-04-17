#!/bin/bash
# P1a on Cardinal H100 (gpu partition). Single-Q per job; env var Q.
# Full config matching Apr-13 pre-fix run for apples-to-apples:
# 30 epochs x 20k train samples x seq=2048 x NoPE x post-fix init.
# train.py saves per-epoch model snapshots so early-stop + eval-from-prior-
# epoch is possible (see checkpoint logic in nanogpt/train.py).
#SBATCH --job-name=needle-nope-2048-h100
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/needle-nope-2048-h100-%j.out
#SBATCH --error=results/needle-nope-2048-h100-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

: "${Q:?Q env var required (e.g. 2.0 or 4.0)}"
SEQ=2048
ARR=$((SEQ - 8))
OUT_DIR="results/needle_stieltjes_q${Q}_seq${SEQ}_nope_initfix_h100"
mkdir -p "${OUT_DIR}"

echo "=== P1a H100: needle stieltjes q=${Q} seq=${SEQ} NoPE post-fix ==="
nvidia-smi
date

python nanogpt/train.py \
    --task needle --attn stieltjes --q "${Q}" \
    --pos-enc none \
    --seq-len "${SEQ}" --max-arr-len "${ARR}" \
    --lr 1e-4 --epochs 30 --batch-size 4 \
    --train-samples 20000 --val-samples 2000 \
    --out "${OUT_DIR}"

# Extrapolation eval — only runs if training finished. If the job is
# scancel'd early, we still have per-epoch ckpts (model_ep001.pt ..
# model_epNN.pt) in OUT_DIR which can be evaluated afterward.
for ES in 2048 4096 8192 16384; do
    if   [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --attn stieltjes --q "${Q}" \
        --seq-len "$ES" --max-arr-len "$((ES - 8))" --val-samples 1000 \
        --batch-size "$EBS" \
        --out "${OUT_DIR}/accuracy_fixed_eval${ES}.json" || echo "FAIL eval ${ES}"
done

date
echo "DONE P1a q=${Q} on H100"
