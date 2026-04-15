#!/bin/bash
#SBATCH --job-name=nanogpt-max-1layer-stj-q4-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-max-1layer-stj-q4-ascend-%j.out
#SBATCH --error=results/nanogpt-max-1layer-stj-q4-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/max_1layer_stieltjes_q4.0_seq2048_nope_ascend"
mkdir -p "${OUT_DIR}"

python -u nanogpt/train.py \
    --task max --max-arr-len 2040 \
    --attn stieltjes --q 4.0 --pos-enc none --seq-len 2048 \
    --lr 3e-4 --epochs 30 --batch-size 4 \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head 6 --n-embd 384 \
    --dtype bf16 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task max --attn stieltjes --q 4.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"
