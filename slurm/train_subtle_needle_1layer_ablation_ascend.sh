#!/bin/bash
# Stieltjes ablation runs: same config as fixedcap, vary one design choice.
# Env vars: ATTN (stieltjes_jt_init | stieltjes_no_safeguard), Q
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

SEQ=128
ARR=120
BS=32

TAG="${ATTN}_q${Q}"
EXTRA="--attn ${ATTN} --q ${Q}"

OUT_DIR="results/subtle_needle_1layer_${TAG}_seq${SEQ}_fixedcap_ascend"
mkdir -p "${OUT_DIR}"

echo "=== ABLATION: ${TAG} seq=${SEQ} ==="

python -u nanogpt/train.py \
    --task needle --needle-margin subtle --max-arr-len "${ARR}" \
    ${EXTRA} \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size "${BS}" \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head 6 --n-embd 384 \
    --dtype bf16 \
    --out "${OUT_DIR}"

# Eval at fixed arr=120 across multiple seqs
for ES in 128 256 512 1024 2048 4096 8192; do
    if   [ "$ES" -le 1024 ]; then EBS=8
    elif [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --needle-margin subtle ${EXTRA} \
        --seq-len "$ES" --max-arr-len "$ARR" --val-samples 1000 \
        --batch-size "$EBS" \
        --out "${OUT_DIR}/accuracy_eval_seq${ES}_arr120.json" || echo "FAIL eval ${ES}"
done

echo "DONE ${OUT_DIR}"
