#!/bin/bash
# Re-run analyze.py on existing model checkpoints to get fresh sample_attn.pt
# (older analyze runs may not have saved sample_attn.pt with the fix)
#
# Run from compute node (needs GPU).
#
# Usage: bash regenerate_analysis.sh [task]
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
    # Extract task name and attn config from dir name
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

    # Filter by task if specified
    if [ -n "$TASK_FILTER" ] && [ "$task" != "$TASK_FILTER" ]; then
        continue
    fi

    # Get seq_len from config.json if available
    seq_len=128
    if [ -f "$dir/config.json" ]; then
        seq_len=$(python -c "import json; print(json.load(open('$dir/config.json')).get('seq_len', 128))")
    fi

    out="$dir/analysis"
    echo "=== Re-analyzing $name (task=$task attn=$attn q=$q seq_len=$seq_len) ==="
    python nanogpt/analyze.py \
        --checkpoint "$model_path" \
        --task "$task" \
        --attn "$attn" \
        --q "$q" \
        --out "$out" \
        --seq-len "$seq_len" || echo "  FAILED for $name"
done
