#!/bin/bash
#SBATCH --job-name=nanogpt-subtle-needle-nope-numiter10-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=results/nanogpt-subtle-needle-nope-numiter10-ascend-%j.out
#SBATCH --error=results/nanogpt-subtle-needle-nope-numiter10-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Coord 0105 probe: does training at num_iter=10 eliminate the
# length-extrapolation failure? If the NR-3 approximation's N-dependent
# output magnitude is what breaks length extrap, converged NR should fix it.
SEQ=2048
ARR=$((SEQ - 8))
Q=4.0
OUT_DIR="results/subtle_needle_stieltjes_q${Q}_seq${SEQ}_numiter10_nope_ascend"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task needle --needle-margin subtle \
    --attn stieltjes --q "${Q}" \
    --pos-enc none \
    --seq-len "${SEQ}" --max-arr-len "${ARR}" \
    --lr 1e-4 --epochs 30 --batch-size 2 \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 10 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task needle --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"
