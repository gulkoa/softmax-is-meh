#!/bin/bash
#SBATCH --job-name=eval-smallseq-32768-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=results/eval-smallseq-32768-ascend-%j.out
#SBATCH --error=results/eval-smallseq-32768-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# 256x extrapolation: eval 6-head seq=128 models at seq=32768.
# Note: softmax at 32768 uses explicit QK^T = 32768^2 = 1B elements/head/batch.
#       6 heads * bf16 * bs=1 = 12GB. Tight but possibly fits in 40GB.
#       stj via Triton is O(N) memory — will fit comfortably.

run_eval() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    local CKPT="$OUT_DIR/model.pt"
    [ -f "$CKPT" ] || CKPT="$OUT_DIR/checkpoint.pt"
    [ -f "$CKPT" ] || { echo "SKIP $OUT_DIR"; return; }
    local OUT_JSON="${OUT_DIR}/accuracy_eval_seq32768_scaledarr.json"
    [ -f "$OUT_JSON" ] && echo "SKIP exists" && return
    echo "=== $(basename $OUT_DIR) eval_seq=32768 ==="
    if [ "$ATTN" = "stieltjes" ]; then
        python nanogpt/eval_accuracy.py \
            --checkpoint "$CKPT" \
            --task needle --needle-margin subtle \
            --attn stieltjes --q "$Q" --use-triton \
            --seq-len 32768 --max-arr-len 32760 \
            --val-samples 500 --batch-size 1 \
            --out "$OUT_JSON" || echo "FAILED"
    else
        python nanogpt/eval_accuracy.py \
            --checkpoint "$CKPT" \
            --task needle --needle-margin subtle --attn softmax \
            --seq-len 32768 --max-arr-len 32760 \
            --val-samples 500 --batch-size 1 \
            --out "$OUT_JSON" || echo "FAILED"
    fi
}

run_eval results/subtle_needle_1layer_softmax_seq128_nope_ascend softmax 1.0
run_eval results/subtle_needle_1layer_stieltjes_q4.0_seq128_nope_ascend stieltjes 4.0
run_eval results/subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend stieltjes 8.0
