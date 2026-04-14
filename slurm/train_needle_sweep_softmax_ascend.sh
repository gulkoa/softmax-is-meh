#!/bin/bash
#SBATCH --job-name=nanogpt-needle-sweep-softmax-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/nanogpt-needle-sweep-softmax-ascend-%j.out
#SBATCH --error=results/nanogpt-needle-sweep-softmax-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for SEQ in 256 512 1024 2048; do
    ARR=$((SEQ - 8))
    OUT_DIR="results/needle_softmax_seq${SEQ}_ascend"
    mkdir -p "${OUT_DIR}"
    echo "=== needle softmax seq_len=${SEQ} ==="
    python nanogpt/train.py \
        --task needle --attn softmax \
        --seq-len "${SEQ}" --max-arr-len "${ARR}" \
        --lr 1e-4 --epochs 30 --batch-size 4 \
        --train-samples 20000 --val-samples 2000 \
        --out "${OUT_DIR}"

    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --attn softmax --q 1.0 \
        --out "${OUT_DIR}/accuracy_fixed.json"
done
