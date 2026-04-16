#!/bin/bash
#SBATCH --job-name=eval-sn1-long-v2-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/eval-sn1-long-v2-ascend-%j.out
#SBATCH --error=results/eval-sn1-long-v2-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Eval-long v2: use --use-triton for stj (flash fwd, O(N) memory)
# and --batch-size 1 for softmax at large seq (explicit QK^T is O(N²))

run_eval() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    local CKPT="$OUT_DIR/checkpoint.pt"
    [ -f "$CKPT" ] || CKPT="$OUT_DIR/model.pt"
    [ -f "$CKPT" ] || { echo "SKIP $OUT_DIR (no checkpoint)"; return; }

    for EVAL_SEQ in 2048 4096 8192; do
        local ARR=$((EVAL_SEQ - 8))
        local OUT_JSON="${OUT_DIR}/accuracy_eval_seq${EVAL_SEQ}_scaledarr.json"
        # Skip if already exists from v1
        [ -f "$OUT_JSON" ] && echo "SKIP $OUT_JSON (exists)" && continue
        echo "=== $(basename $OUT_DIR) eval_seq=${EVAL_SEQ} max_arr=${ARR} ==="
        if [ "$ATTN" = "stieltjes" ]; then
            # Use Triton flash forward (correct, O(N) memory) for long evals
            python nanogpt/eval_accuracy.py \
                --checkpoint "$CKPT" \
                --task needle --needle-margin subtle \
                --attn stieltjes --q "$Q" --use-triton \
                --seq-len "$EVAL_SEQ" --max-arr-len "$ARR" \
                --val-samples 2000 --batch-size 1 \
                --out "$OUT_JSON" || echo "FAILED: $OUT_JSON"
        else
            # Softmax: reduce batch size to fit O(N²) attention
            python nanogpt/eval_accuracy.py \
                --checkpoint "$CKPT" \
                --task needle --needle-margin subtle \
                --attn softmax \
                --seq-len "$EVAL_SEQ" --max-arr-len "$ARR" \
                --val-samples 2000 --batch-size 1 \
                --out "$OUT_JSON" || echo "FAILED: $OUT_JSON"
        fi
    done
}

run_eval results/subtle_needle_1layer_softmax_seq2048_nope_ascend softmax 1.0
run_eval results/subtle_needle_1layer_stieltjes_q8.0_seq2048_nope_ascend stieltjes 8.0
run_eval results/subtle_needle_1layer_stieltjes_q16.0_seq2048_nope_ascend stieltjes 16.0
run_eval results/subtle_needle_1layer_stieltjes_q32.0_seq2048_nope_ascend stieltjes 32.0
run_eval results/subtle_needle_1layer_stieltjes_q64.0_seq2048_nope_ascend stieltjes 64.0
