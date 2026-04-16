#!/bin/bash
#SBATCH --job-name=max-first1
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/max-first1-%j.out
#SBATCH --error=results/max-first1-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

ARR=120
NEMBD=32
NHEAD=4

run_eval() {
    local OUT_DIR="$1"
    local ATTN="$2"
    local Q="$3"
    [ -d "$OUT_DIR" ] || { echo "SKIP missing $OUT_DIR"; return; }
    for ES in 128 256 512 1024 2048 4096 8192; do
        if   [ "$ES" -le 1024 ]; then EBS=8
        elif [ "$ES" -le 4096 ]; then EBS=2
        else                          EBS=1
        fi
        python nanogpt/eval_accuracy.py \
            --checkpoint "${OUT_DIR}/model.pt" \
            --task max \
            --attn "$ATTN" --q "$Q" \
            --seq-len "$ES" --max-arr-len "$ARR" --val-samples 1000 \
            --batch-size "$EBS" \
            --n-layer 1 --n-head ${NHEAD} --n-embd ${NEMBD} \
            --first-n-output 1 \
            --out "${OUT_DIR}/accuracy_eval_seq${ES}_arr120_first1.json" || echo "FAIL ${ES}"
    done
}

run_eval results/max_1layer_softmax_seq128_embd32_h4_initfix_ascend       softmax   1.0
for q in 1.0 2.0 4.0 8.0 16.0 32.0; do
    run_eval results/max_1layer_stieltjes_q${q}_seq128_embd32_h4_initfix_ascend stieltjes $q
done
echo "DONE"
