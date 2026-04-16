#!/bin/bash
# P1b: binary_search 6-layer stieltjes q=8 AND q=16 with POST-FIX init (lambda_0=1.1).
# Verifies the "q-cliff at 6-layer is initialization-independent" claim in Table 1.
# Pre-fix results: q=8 seed=42: 0.413, q=16 seed=42: <=0.05. Claim stands if post-fix
# values also stay low (<0.5). If either reaches >0.85, PING COORDINATOR IMMEDIATELY.
# Env var: Q (8.0 or 16.0)
#SBATCH --job-name=bsearch-highq-initfix
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/bsearch-highq-initfix-%j.out
#SBATCH --error=results/bsearch-highq-initfix-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/binary_search_stieltjes_q${Q}_initfix_ascend"
mkdir -p "${OUT_DIR}"
echo "=== P1b: binary_search 6-layer stieltjes q=${Q} post-fix init ==="

python nanogpt/train.py \
    --task binary_search --attn stieltjes --q "${Q}" \
    --epochs 30 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task binary_search --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"

echo "DONE P1b q=${Q}"
