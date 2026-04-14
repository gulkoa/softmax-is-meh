#!/bin/bash
#SBATCH --job-name=nanogpt-needle-nope-seq4096-stj-q4-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=results/nanogpt-needle-nope-seq4096-stj-q4-ascend-%j.out
#SBATCH --error=results/nanogpt-needle-nope-seq4096-stj-q4-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=4096
ARR=$((SEQ - 8))
Q=4.0
OUT_DIR="results/needle_stieltjes_q${Q}_seq${SEQ}_nope_ascend"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task needle --attn stieltjes --q "${Q}" \
    --pos-enc none \
    --seq-len "${SEQ}" --max-arr-len "${ARR}" \
    --lr 1e-4 --epochs 30 --batch-size 1 \
    --train-samples 10000 --val-samples 1000 \
    --stieltjes-num-iter 3 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task needle --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"
