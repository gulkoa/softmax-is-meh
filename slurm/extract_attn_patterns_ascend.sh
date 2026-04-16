#!/bin/bash
#SBATCH --job-name=extract-attn-patterns-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=results/extract-attn-patterns-ascend-%j.out
#SBATCH --error=results/extract-attn-patterns-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

run_extract() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    [ -d "$OUT_DIR" ] || { echo "SKIP missing $OUT_DIR"; return; }
    python figures/extract_attention_patterns.py \
        --model "$OUT_DIR" \
        --attn "$ATTN" --q "$Q" \
        --eval-seqs 128 512 2048 8192 \
        --num-samples 32 \
        --needle-margin subtle || echo "FAILED $OUT_DIR"
}

run_extract results/subtle_needle_1layer_softmax_seq128_nope_ascend softmax 1.0
run_extract results/subtle_needle_1layer_stieltjes_q4.0_seq128_nope_ascend stieltjes 4.0
run_extract results/subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend stieltjes 8.0
run_extract results/subtle_needle_1layer_stieltjes_q16.0_seq128_nope_ascend stieltjes 16.0
run_extract results/subtle_needle_1layer_stieltjes_q24.0_seq128_nope_ascend stieltjes 24.0
run_extract results/subtle_needle_1layer_stieltjes_q32.0_seq128_nope_ascend stieltjes 32.0
