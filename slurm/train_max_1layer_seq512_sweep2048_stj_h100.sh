#!/bin/bash
#SBATCH --job-name=nanogpt-max-1l-seq512-sweep-stj-h100
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=2:00:00
#SBATCH --output=results/nanogpt-max-1l-seq512-sweep-stj-h100-%j.out
#SBATCH --error=results/nanogpt-max-1l-seq512-sweep-stj-h100-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

TRAIN_SEQ=512
OUT_DIR="results/max_1layer_stieltjes_q4.0_seq${TRAIN_SEQ}_nope_h100"
mkdir -p "${OUT_DIR}"

echo "=== max 1-layer stj q=4 train-seq=${TRAIN_SEQ} NoPE h100 ==="
nvidia-smi

python -u nanogpt/train.py \
    --task max --max-arr-len $((TRAIN_SEQ - 8)) \
    --attn stieltjes --q 4.0 \
    --pos-enc none \
    --seq-len "${TRAIN_SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size 8 \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head 6 --n-embd 384 \
    --dtype bf16 \
    --out "${OUT_DIR}"

# Eval-long sweep at 512, 1024, 2048
for EVAL_SEQ in 512 1024 2048; do
    echo "=== eval at seq=${EVAL_SEQ} ==="
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task max --attn stieltjes --q 4.0 \
        --seq-len "${EVAL_SEQ}" --val-samples 2000 \
        --out "${OUT_DIR}/accuracy_eval_seq${EVAL_SEQ}.json"
done
