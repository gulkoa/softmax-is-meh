#!/bin/bash
#SBATCH --job-name=eval-fixedcap-extreme
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --output=results/eval-fixedcap-extreme-%j.out
#SBATCH --error=results/eval-fixedcap-extreme-%j.err

# Extreme OOD (seq=16384, 32768) eval on the fresh fixedcap retrains.
# Tests whether softmax finally disperses at 128x/256x extrap; quantifies
# stj's behavior at extreme OOD under the corrected task.

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

ARR=120

run_extreme() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    local CKPT="$OUT_DIR/model.pt"
    [ -f "$CKPT" ] || { echo "SKIP $OUT_DIR (no ckpt)"; return; }
    for ES in 16384 32768; do
        OUT_JSON="${OUT_DIR}/accuracy_eval_seq${ES}_arr120.json"
        TRITON_FLAG=""
        [ "$ATTN" = "stieltjes" ] && TRITON_FLAG="--use-triton"
        echo "=== $(basename $OUT_DIR) eval_seq=${ES} arr=${ARR} ${TRITON_FLAG} ==="
        python nanogpt/eval_accuracy.py \
            --checkpoint "$CKPT" \
            --task needle --needle-margin subtle \
            --attn "$ATTN" --q "$Q" \
            --seq-len "$ES" --max-arr-len "$ARR" \
            --val-samples 500 --batch-size 1 $TRITON_FLAG \
            --out "$OUT_JSON" || echo "FAIL $OUT_JSON"
    done
}

run_extreme results/subtle_needle_1layer_softmax_seq128_fixedcap_ascend       softmax   1.0
run_extreme results/subtle_needle_1layer_stieltjes_q4.0_seq128_fixedcap_ascend  stieltjes 4.0
run_extreme results/subtle_needle_1layer_stieltjes_q8.0_seq128_fixedcap_ascend  stieltjes 8.0
run_extreme results/subtle_needle_1layer_stieltjes_q16.0_seq128_fixedcap_ascend stieltjes 16.0

# Triton-off control on the fresh stj checkpoints
run_ref() {
    local OUT_DIR="$1"
    local Q="$2"
    local CKPT="$OUT_DIR/model.pt"
    [ -f "$CKPT" ] || return
    for ES in 2048 4096; do
        OUT_JSON="${OUT_DIR}/accuracy_eval_seq${ES}_arr120_ref.json"
        echo "=== REF $(basename $OUT_DIR) eval_seq=${ES} arr=${ARR} (no Triton) ==="
        python nanogpt/eval_accuracy.py \
            --checkpoint "$CKPT" \
            --task needle --needle-margin subtle \
            --attn stieltjes --q "$Q" \
            --seq-len "$ES" --max-arr-len "$ARR" \
            --val-samples 500 --batch-size 2 \
            --out "$OUT_JSON" || echo "FAIL $OUT_JSON"
    done
}

run_ref results/subtle_needle_1layer_stieltjes_q4.0_seq128_fixedcap_ascend  4.0
run_ref results/subtle_needle_1layer_stieltjes_q8.0_seq128_fixedcap_ascend  8.0
run_ref results/subtle_needle_1layer_stieltjes_q16.0_seq128_fixedcap_ascend 16.0

echo "DONE"
