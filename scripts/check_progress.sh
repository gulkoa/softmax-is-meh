#!/bin/bash
# Check training progress across all experiment runs
RESULTS_DIR="${1:-/users/PAS2402/alexg/softmax/softmax-is-meh/results}"

echo "=== Training Progress ==="
echo ""
printf "%-45s %6s %8s\n" "Run" "Epoch" "Accuracy"
printf "%-45s %6s %8s\n" "---" "-----" "--------"

for d in "$RESULTS_DIR"/*/; do
    metrics="$d/metrics.csv"
    if [ -f "$metrics" ]; then
        name=$(basename "$d")
        epochs=$(( $(wc -l < "$metrics") - 1 ))
        if [ "$epochs" -gt 0 ]; then
            acc=$(tail -1 "$metrics" | cut -d, -f4)
            printf "%-45s %6d %8s\n" "$name" "$epochs" "$acc"
        fi
    fi
done

echo ""
echo "=== Queue ==="
squeue -u "$USER" -l 2>/dev/null || echo "(not on SLURM node)"
