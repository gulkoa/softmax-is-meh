#!/bin/bash
#SBATCH --job-name=velickovic-entropy-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=results/velickovic-entropy-ascend-%j.out
#SBATCH --error=results/velickovic-entropy-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Attention entropy + std diagnostic for Velickovic replication.
# Runs on trained seq=128 subtle-needle checkpoints at eval_seq ∈ {128, 512, 2048, 8192}.

run_analyze() {
    local OUT_DIR="$1"  # model dir
    local ATTN="$2"
    local Q="$3"
    local CKPT="$OUT_DIR/model.pt"
    [ -f "$CKPT" ] || CKPT="$OUT_DIR/checkpoint.pt"
    [ -f "$CKPT" ] || { echo "SKIP $OUT_DIR"; return; }

    for ES_N in "128 100" "512 50" "2048 16"; do
        EVAL_SEQ=$(echo $ES_N | cut -d' ' -f1)
        NSAMP=$(echo $ES_N | cut -d' ' -f2)
        ARR=$((EVAL_SEQ - 8))
        ANALYSIS_DIR="${OUT_DIR}/entropy_seq${EVAL_SEQ}"
        mkdir -p "$ANALYSIS_DIR"
        echo "=== $(basename $OUT_DIR) eval_seq=${EVAL_SEQ} samples=${NSAMP} ==="
        python nanogpt/analyze.py \
            --checkpoint "$CKPT" \
            --task needle --needle-margin subtle \
            --attn "$ATTN" --q "$Q" \
            --seq-len "$EVAL_SEQ" --max-arr-len "$ARR" \
            --num-samples "$NSAMP" \
            --out "$ANALYSIS_DIR" || echo "FAILED $ANALYSIS_DIR"
    done
}

# 6-head subtle-needle seq=128 models (Velickovic sweep A)
run_analyze results/subtle_needle_1layer_softmax_seq128_nope_ascend softmax 1.0
run_analyze results/subtle_needle_1layer_stieltjes_q4.0_seq128_nope_ascend stieltjes 4.0
run_analyze results/subtle_needle_1layer_stieltjes_q8.0_seq128_nope_ascend stieltjes 8.0
run_analyze results/subtle_needle_1layer_stieltjes_q16.0_seq128_nope_ascend stieltjes 16.0
run_analyze results/subtle_needle_1layer_stieltjes_q24.0_seq128_nope_ascend stieltjes 24.0
run_analyze results/subtle_needle_1layer_stieltjes_q32.0_seq128_nope_ascend stieltjes 32.0
