#!/bin/bash
# Ascend sibling of regenerate_analysis.sh. Writes to `analysis_ascend/` instead
# of `analysis/` so it doesn't collide with Cardinal's queued regen-analysis
# jobs (8521356, 8526227) that target the same checkpoints.
#
# Run from Ascend compute node (needs GPU).
#
# Usage: bash regenerate_analysis_ascend.sh [task]
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

TASK_FILTER="${1:-}"

for dir in results/*/; do
    name=$(basename "$dir")
    model_path="$dir/model.pt"
    if [ ! -f "$model_path" ]; then
        continue
    fi
    if [[ "$name" =~ ^([a-z_]+)_softmax ]]; then
        task="${BASH_REMATCH[1]}"
        attn="softmax"
        q="1.0"
    elif [[ "$name" =~ ^([a-z_]+)_stieltjes_q([0-9.]+) ]]; then
        task="${BASH_REMATCH[1]}"
        attn="stieltjes"
        q="${BASH_REMATCH[2]}"
    else
        continue
    fi

    if [ -n "$TASK_FILTER" ] && [ "$task" != "$TASK_FILTER" ]; then
        continue
    fi

    seq_len=128
    if [ -f "$dir/config.json" ]; then
        seq_len=$(python -c "import json; print(json.load(open('$dir/config.json')).get('seq_len', 128))")
    fi

    out="$dir/analysis_ascend"
    echo "=== Re-analyzing $name (task=$task attn=$attn q=$q seq_len=$seq_len) -> $out ==="
    python nanogpt/analyze.py \
        --checkpoint "$model_path" \
        --task "$task" \
        --attn "$attn" \
        --q "$q" \
        --out "$out" \
        --seq-len "$seq_len" || echo "  FAILED for $name"
done
