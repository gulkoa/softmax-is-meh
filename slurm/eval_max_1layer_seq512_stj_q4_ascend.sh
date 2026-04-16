#!/bin/bash
#SBATCH --job-name=eval-max-1layer-seq512-stj-q4-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/eval-max-1layer-seq512-stj-q4-ascend-%j.out
#SBATCH --error=results/eval-max-1layer-seq512-stj-q4-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/max_1layer_stieltjes_q4.0_seq512_nope_ascend"

for S in 512 1024 2048; do
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task max --attn stieltjes --q 4.0 \
        --seq-len "$S" --val-samples 2000 \
        --out "${OUT_DIR}/accuracy_eval_seq${S}.json"
done
