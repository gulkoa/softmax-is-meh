#!/bin/bash
#SBATCH --job-name=nanogpt-needle-nope-q4-seq2048-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=results/nanogpt-needle-nope-q4-seq2048-ascend-%j.out
#SBATCH --error=results/nanogpt-needle-nope-q4-seq2048-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Standalone q=4 NoPE retrain. 4905959 is sequential (softmax then q=4) and
# will time out before reaching q=4, so we submit q=4 on its own here.
SEQ=2048
ARR=$((SEQ - 8))
Q=4.0
OUT_DIR="results/needle_stieltjes_q${Q}_seq${SEQ}_nope_ascend"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task needle --attn stieltjes --q "${Q}" \
    --pos-enc none \
    --seq-len "${SEQ}" --max-arr-len "${ARR}" \
    --lr 1e-4 --epochs 30 --batch-size 4 \
    --train-samples 20000 --val-samples 2000 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task needle --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"
