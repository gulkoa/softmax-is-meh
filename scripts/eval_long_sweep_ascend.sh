#!/bin/bash
# Eval-long sweep: take the NoPE needle checkpoints (trained at seq=2048)
# and evaluate at progressively longer seq_lens via the Triton fwd kernel.
#
# Designed to run on an Ascend GPU node (login node won't work — needs CUDA).
# For very long sequences (16k, 32k) this depends on the Triton kernel being
# memory-efficient (flash-style) so the attn matrix fits.
#
# Outputs: <ckpt_dir>/accuracy_fixed_eval{SEQ}.json per (checkpoint, seq) pair.
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

EVAL_SEQS=${EVAL_SEQS:-"2048 4096 8192 16384 32768"}
VAL_SAMPLES=${VAL_SAMPLES:-500}

CKPTS=(
    "results/needle_softmax_q1.0_seq2048_nope_ascend"
    "results/needle_stieltjes_q4.0_seq2048_nope_ascend"
)

for CKPT_DIR in "${CKPTS[@]}"; do
    # Prefer model.pt (end-of-training save); fall back to checkpoint.pt (every-10-epoch save)
    # for jobs that hit walltime before the final save.
    if [ -f "${CKPT_DIR}/model.pt" ]; then
        CKPT_FILE="${CKPT_DIR}/model.pt"
    elif [ -f "${CKPT_DIR}/checkpoint.pt" ]; then
        CKPT_FILE="${CKPT_DIR}/checkpoint.pt"
    else
        echo "SKIP ${CKPT_DIR}: no model.pt or checkpoint.pt yet"
        continue
    fi
    # Read attn / q from config.json
    ATTN=$(python -c "import json; print(json.load(open('${CKPT_DIR}/config.json'))['attn'])")
    Q=$(python -c "import json; print(json.load(open('${CKPT_DIR}/config.json'))['q'])")

    for SEQ in $EVAL_SEQS; do
        ARR=$((SEQ - 8))
        OUT_JSON="${CKPT_DIR}/accuracy_fixed_eval${SEQ}.json"
        # Shrink batch_size as N grows: naive softmax attn uses O(B*H*N^2) memory,
        # so at N=32768 on A100-40GB softmax only fits at B=1 (and even then barely
        # — 32k is expected to OOM on softmax; Triton kernel handles it fine).
        if [ "${SEQ}" -le 2048 ]; then
            BS=4
        elif [ "${SEQ}" -le 4096 ]; then
            BS=2
        else
            BS=1
        fi
        echo "=== eval ${CKPT_DIR} (${CKPT_FILE##*/}) at seq=${SEQ} bs=${BS} ==="
        python nanogpt/eval_accuracy.py \
            --checkpoint "${CKPT_FILE}" \
            --task needle --attn "${ATTN}" --q "${Q}" \
            --eval-seq-len "${SEQ}" --max-arr-len "${ARR}" \
            --val-samples "${VAL_SAMPLES}" --batch-size "${BS}" \
            ${USE_TRITON:+--use-triton} \
            --out "${OUT_JSON}" || echo "  FAILED at seq=${SEQ}"
    done
done

echo "=== summary ==="
python scripts/needle_results.py
