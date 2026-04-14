#!/bin/bash
#SBATCH --job-name=nanogpt-needle-nope-seq4096-softmax-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=results/nanogpt-needle-nope-seq4096-softmax-ascend-%j.out
#SBATCH --error=results/nanogpt-needle-nope-seq4096-softmax-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Native seq=4096 needle training. bs=2 to keep B*N^2 ~ constant vs seq=2048 bs=4.
# 10k train samples (half of the seq=2048 run) to stay within 10h walltime
# given ~2x per-epoch wall at 2x seq_len.
SEQ=4096
ARR=$((SEQ - 8))
OUT_DIR="results/needle_softmax_q1.0_seq${SEQ}_nope_ascend"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task needle --attn softmax \
    --pos-enc none \
    --seq-len "${SEQ}" --max-arr-len "${ARR}" \
    --lr 1e-4 --epochs 30 --batch-size 2 \
    --train-samples 10000 --val-samples 1000 \
    --stieltjes-num-iter 3 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task needle --attn softmax --q 1.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"
