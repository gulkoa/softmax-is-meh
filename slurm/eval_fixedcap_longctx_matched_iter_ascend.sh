#!/bin/bash
# Eval fixedcap corrected-init checkpoints at long context using PyTorch ref
# (no Triton) so num_iter matches training (3). Fixes the Triton num_iter
# mismatch that caused spurious collapse at seq>=2048 in the original eval.
# Env vars: ATTN, Q
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

ARR=120

if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA_EVAL="--attn stieltjes --q ${Q}"
else
    TAG="softmax"
    EXTRA_EVAL="--attn softmax --q 1.0"
fi

CKPT_DIR="results/subtle_needle_1layer_${TAG}_seq128_fixedcap_ascend"

if [ ! -f "${CKPT_DIR}/model.pt" ]; then
    echo "ERROR: checkpoint not found at ${CKPT_DIR}/model.pt"
    exit 1
fi

echo "=== Long-ctx eval (matched num_iter=3, no Triton): ${TAG} ==="

for ES in 256 512 1024 2048 4096 8192; do
    if   [ "$ES" -le 1024 ]; then EBS=8
    elif [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    python nanogpt/eval_accuracy.py \
        --checkpoint "${CKPT_DIR}/model.pt" \
        --task needle --needle-margin subtle ${EXTRA_EVAL} \
        --seq-len "$ES" --max-arr-len "$ARR" --val-samples 1000 \
        --batch-size "$EBS" \
        --num-iter-override 3 \
        --out "${CKPT_DIR}/accuracy_eval_seq${ES}_arr120_ni3.json" || echo "FAIL eval ${ES}"
done

echo "DONE"
