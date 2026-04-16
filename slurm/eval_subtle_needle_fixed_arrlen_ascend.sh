#!/bin/bash
#SBATCH --job-name=eval-fixed-arrlen
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=results/eval-fixed-arrlen-%j.out
#SBATCH --error=results/eval-fixed-arrlen-%j.err

# Re-eval Velickovic-replication seq=128-trained subtle-needle 1-layer models
# with FIXED max_arr_len=120 at increasing eval_seq.
#
# Why: scaledarr eval is degenerate at seq>>max_val=64 because needle = max(arr)+1 = 65
# is constant for any large array → "always predict 65" model scores 1.0.
# Fixed max_arr_len keeps task difficulty constant; only the padded context length grows.
# This is the clean length-extrapolation test.

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

ARR=120  # fixed: same task at every eval_seq
EVAL_SEQS="128 256 512 1024 2048 4096 8192"

run_eval() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    local CKPT="$OUT_DIR/model.pt"
    [ -f "$CKPT" ] || { echo "SKIP $OUT_DIR (no ckpt)"; return; }
    for ES in $EVAL_SEQS; do
        OUT_JSON="${OUT_DIR}/accuracy_eval_seq${ES}_arr120.json"
        # Adaptive batch size: drop with seq^2 to fit O(N^2) attention memory
        if   [ "$ES" -le 1024 ]; then BS=8
        elif [ "$ES" -le 4096 ]; then BS=2
        else                          BS=1
        fi
        # Use Triton (O(N) memory) for stj at seq>=2048 to allow longer seq
        TRITON_FLAG=""
        if [ "$ATTN" = "stieltjes" ] && [ "$ES" -ge 2048 ]; then
            TRITON_FLAG="--use-triton"
        fi
        echo "=== $(basename $OUT_DIR) eval_seq=${ES} arr=${ARR} bs=${BS} ${TRITON_FLAG} ==="
        python nanogpt/eval_accuracy.py \
            --checkpoint "$CKPT" \
            --task needle --needle-margin subtle \
            --attn "$ATTN" --q "$Q" \
            --seq-len "$ES" --max-arr-len "$ARR" \
            --val-samples 1000 \
            --batch-size "$BS" $TRITON_FLAG \
            --out "$OUT_JSON" || echo "FAIL $OUT_JSON"
    done
}

# Sweep A: 6-head subtle-needle seq=128 trained models
run_eval results/subtle_needle_1layer_softmax_seq128_nope_ascend       softmax   1.0
run_eval results/subtle_needle_1layer_stieltjes_q4.0_seq128_nope_ascend  stieltjes 4.0
run_eval results/subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend  stieltjes 8.0
run_eval results/subtle_needle_1layer_stieltjes_q16.0_seq128_nope_ascend stieltjes 16.0
run_eval results/subtle_needle_1layer_stieltjes_q24.0_seq128_nope_ascend stieltjes 24.0
run_eval results/subtle_needle_1layer_stieltjes_q32.0_seq128_nope_ascend stieltjes 32.0
[ -d results/subtle_needle_1layer_stieltjes_q64.0_seq128_nope_ascend ] && \
  run_eval results/subtle_needle_1layer_stieltjes_q64.0_seq128_nope_ascend stieltjes 64.0

echo "DONE"
