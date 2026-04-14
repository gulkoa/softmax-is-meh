#!/bin/bash
# Executor babysitter: compact view of queue + in-flight retrains + finished
# accuracy_fixed.json files. Safe to run from anywhere; paths are absolute.
set -euo pipefail

REPO="/users/PAS2402/alexg/softmax/softmax-is-meh"
cd "$REPO"

echo "=== SQUEUE (alexg) ==="
squeue -u alexg 2>&1

echo
echo "=== LATEST METRICS.CSV TAILS (2 lines) ==="
for d in results/*_ascend_retrain/; do
    f="${d}metrics.csv"
    [ -f "$f" ] || continue
    last=$(tail -1 "$f")
    [ -n "$last" ] || continue
    printf "  %-70s  %s\n" "$(basename "$d")" "$last"
done | sort

echo
echo "=== FINISHED accuracy_fixed.json ==="
for j in results/*_ascend_retrain/accuracy_fixed.json; do
    [ -f "$j" ] || continue
    python -c "
import json, sys
d = json.load(open('$j'))
print(f\"  {d['checkpoint'].split('/')[-2]:<60s} \"
      f\"fixed={d['accuracy_fixed']:.4f}  all={d['accuracy_all']:.4f}  \"
      f\"echo={d['accuracy_input_echo']:.4f}\")
"
done | sort
