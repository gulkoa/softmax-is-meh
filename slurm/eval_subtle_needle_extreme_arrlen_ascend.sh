#!/bin/bash
#SBATCH --job-name=eval-extreme-arrlen
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=results/eval-extreme-arrlen-%j.out
#SBATCH --error=results/eval-extreme-arrlen-%j.err

# Extreme-OOD fixed-arr-len eval (seq=16384, 32768) + Triton-vs-ref sanity at seq=2048/4096.
# Goal 1: push softmax past 64x to see if Velickovic dispersion finally triggers.
# Goal 2: rule out Triton numerical issue as cause of stj's U-shape at seq=2048/4096.

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

ARR=120

# --- Goal 1: extreme OOD ---
# softmax: O(N^2) attention. seq=16384 needs ~3.2GB; seq=32768 ~12.9GB. bs=1 fits 40GB A100.
# stj: use Triton (O(N) memory).
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
        echo "=== EXTREME $(basename $OUT_DIR) eval_seq=${ES} arr=${ARR} ${TRITON_FLAG} ==="
        python nanogpt/eval_accuracy.py \
            --checkpoint "$CKPT" \
            --task needle --needle-margin subtle \
            --attn "$ATTN" --q "$Q" \
            --seq-len "$ES" --max-arr-len "$ARR" \
            --val-samples 500 --batch-size 1 $TRITON_FLAG \
            --out "$OUT_JSON" || echo "FAIL $OUT_JSON"
    done
}

run_extreme results/subtle_needle_1layer_softmax_seq128_nope_ascend       softmax   1.0
run_extreme results/subtle_needle_1layer_stieltjes_q4.0_seq128_nope_ascend  stieltjes 4.0
run_extreme results/subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend  stieltjes 8.0
run_extreme results/subtle_needle_1layer_stieltjes_q16.0_seq128_nope_ascend stieltjes 16.0

# --- Goal 2: stj WITHOUT --use-triton at seq=2048/4096 (verify U-shape) ---
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

run_ref results/subtle_needle_1layer_stieltjes_q4.0_seq128_nope_ascend  4.0
run_ref results/subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend  8.0
run_ref results/subtle_needle_1layer_stieltjes_q16.0_seq128_nope_ascend 16.0

echo "DONE"
