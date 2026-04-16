#!/bin/bash
#SBATCH --job-name=eval-max-1layer-scaledarr-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=results/eval-max-1layer-scaledarr-ascend-%j.out
#SBATCH --error=results/eval-max-1layer-scaledarr-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

run_eval() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    for S in 512 1024 2048; do
        local ARR=$((S - 8))
        local OUT="${OUT_DIR}/accuracy_eval_seq${S}_scaledarr.json"
        echo "=== ${OUT_DIR} attn=${ATTN} q=${Q} seq=${S} max_arr=${ARR} ==="
        if [ "$ATTN" = "stieltjes" ]; then
            python nanogpt/eval_accuracy.py \
                --checkpoint "${OUT_DIR}/model.pt" \
                --task max --attn stieltjes --q "$Q" \
                --seq-len "$S" --max-arr-len "$ARR" --val-samples 2000 \
                --out "$OUT"
        else
            python nanogpt/eval_accuracy.py \
                --checkpoint "${OUT_DIR}/model.pt" \
                --task max --attn softmax \
                --seq-len "$S" --max-arr-len "$ARR" --val-samples 2000 \
                --out "$OUT"
        fi
    done
}

run_eval results/max_1layer_stieltjes_q4.0_seq512_nope_ascend stieltjes 4.0
run_eval results/max_1layer_softmax_seq512_nope_ascend softmax 1.0
run_eval results/max_1layer_stieltjes_q24.0_seq512_nope_ascend stieltjes 24.0
