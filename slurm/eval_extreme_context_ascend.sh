#!/bin/bash
#SBATCH --job-name=eval-extremectx
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=results/eval-extremectx-%j.out
#SBATCH --error=results/eval-extremectx-%j.err

# Push softmax + stj to extreme context (32k -> 131k = 1024x extrap)
# using SDPA flash for softmax (O(N) memory) and Triton for stj.
# Hypothesis: at extreme N softmax should finally disperse (Velickovic Theorem 2.2)
# while stj's polynomial decay preserves needle weight.

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

EVAL_SEQS="32768 65536 131072"

run_extreme() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    [ -d "$OUT_DIR" ] || { echo "SKIP missing $OUT_DIR"; return; }
    python figures/eval_extreme_context.py \
        --model "$OUT_DIR" \
        --attn "$ATTN" --q "$Q" \
        --eval-seqs $EVAL_SEQS \
        --val-samples 200 --batch-size 1 \
        --needle-margin subtle || echo "FAILED $OUT_DIR"
}

run_extreme results/subtle_needle_1layer_softmax_seq128_fixedcap_ascend       softmax   1.0
run_extreme results/subtle_needle_1layer_stieltjes_q4.0_seq128_fixedcap_ascend  stieltjes 4.0
run_extreme results/subtle_needle_1layer_stieltjes_q8.0_seq128_fixedcap_ascend  stieltjes 8.0

echo "DONE"
