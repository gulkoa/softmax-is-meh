#!/bin/bash
# Velickovic-strict regime: train at very small N, eval at extreme N.
# If softmax disperses anywhere, this is where.
# Env vars: ATTN, Q, NTRAIN
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=${NTRAIN:-32}
ARR=$((SEQ - 8))
BS=64

if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA_TRAIN="--attn stieltjes --q ${Q}"
    EXTRA_EVAL="--attn stieltjes --q ${Q}"
else
    TAG="softmax"
    EXTRA_TRAIN="--attn softmax"
    EXTRA_EVAL="--attn softmax --q 1.0"
fi

OUT_DIR="results/subtle_needle_1layer_${TAG}_seq${SEQ}_velickovic_ascend"
mkdir -p "${OUT_DIR}"

echo "=== VELICKOVIC-STRICT: subtle-needle 1-layer ${TAG} seq=${SEQ} ==="

python -u nanogpt/train.py \
    --task needle --needle-margin subtle --max-arr-len "${ARR}" \
    ${EXTRA_TRAIN} \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 3e-4 --epochs 50 --batch-size "${BS}" \
    --train-samples 50000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head 6 --n-embd 384 \
    --dtype bf16 \
    --out "${OUT_DIR}"

# Eval at extreme OOD: 256x, 512x, 1024x extrapolation
for ES in $SEQ $((SEQ * 8)) $((SEQ * 64)) $((SEQ * 256)) $((SEQ * 1024)); do
    if [ "$ES" -gt 32768 ]; then continue; fi
    if   [ "$ES" -le 1024 ]; then EBS=8
    elif [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    TRITON_FLAG=""
    [ "$ATTN" = "stieltjes" ] && [ "$ES" -ge 2048 ] && TRITON_FLAG="--use-triton"
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --needle-margin subtle ${EXTRA_EVAL} \
        --seq-len "$ES" --max-arr-len "$ARR" --val-samples 1000 \
        --batch-size "$EBS" $TRITON_FLAG \
        --out "${OUT_DIR}/accuracy_eval_seq${ES}_arr${ARR}.json" || echo "FAIL eval ${ES}"
done

echo "DONE ${OUT_DIR}"
