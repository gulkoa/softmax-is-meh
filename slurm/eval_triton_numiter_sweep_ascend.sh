#!/bin/bash
#SBATCH --job-name=triton-numiter
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/triton-numiter-%j.out
#SBATCH --error=results/triton-numiter-%j.err

# Test if Triton U-shape at intermediate seq is fixable by more NR iterations.
# Default Triton uses num_iter=10. Test 3, 10, 20, 50 on stj q=4 fixedcap at
# seq=2048 (where Triton gave 0.027, ref gives 0.976).

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

CKPT_DIR="results/subtle_needle_1layer_stieltjes_q4.0_seq128_fixedcap_ascend"

for NI in 3 10 20 50; do
    OUT_JSON="${CKPT_DIR}/accuracy_eval_seq2048_arr120_triton_ni${NI}.json"
    echo "=== Triton num_iter=${NI} at seq=2048 ==="
    python nanogpt/eval_accuracy.py \
        --checkpoint "${CKPT_DIR}/model.pt" \
        --task needle --needle-margin subtle \
        --attn stieltjes --q 4.0 \
        --seq-len 2048 --max-arr-len 120 --val-samples 500 \
        --batch-size 2 --use-triton \
        --num-iter-override "$NI" \
        --out "$OUT_JSON" || echo "FAIL ni=${NI}"
done

echo "DONE"
