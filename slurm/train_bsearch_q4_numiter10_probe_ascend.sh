#!/bin/bash
#SBATCH --job-name=nanogpt-bsearch-q4-numiter10-probe-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-bsearch-q4-numiter10-probe-ascend-%j.out
#SBATCH --error=results/nanogpt-bsearch-q4-numiter10-probe-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Path C probe: train binary_search q=4 at stieltjes_num_iter=10 instead of
# default 3. If the resulting model hits comparable accuracy to the
# num_iter=3 model (which got 0.9996), then models are num_iter-robust and
# the workshop paper gets a one-sentence "num_iter does not matter much in
# the trained-model regime" note. If it trains to substantially different
# accuracy, we have a real "NR iteration count as implicit regularizer"
# finding.
OUT_DIR="results/binary_search_stieltjes_q4.0_numiter10_ascend_retrain"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task binary_search --attn stieltjes --q 4.0 \
    --stieltjes-num-iter 10 \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task binary_search --attn stieltjes --q 4.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"
