#!/bin/bash
# Run the pre-registered nan check (scripts/early_cancel_nan.py --cancel) on
# the 1e-3 contrast runs every 5 minutes until the three A chains have
# drained (sacct shows every job of theirs in a terminal state). Written
# 2026-10-27 to replace the session-scratchpad loop lost with that temp
# directory. Prints the checker's lines only when they change from the
# previous pass, so a quiet run stays quiet.
#   bash scripts/contrast_nan_loop.sh
set -u
ROOT=/users/PAS2402/alexg/softmax/softmax-is-meh
PY=$ROOT/triton/.venv/bin/python
JOBS=$ROOT/results/contrast_1e3_seeds_jobs.txt
IDS=15669636,15669637,15669638,15669639,15669640,15669641,15669642,15669643,15669644,15669645,15669646,15669647,15669648,15669649,15669650
TERMINAL="COMPLETED|TIMEOUT|FAILED|CANCELLED|NODE_FAIL|OUT_OF_ME|PREEMPTED|DEADLINE"
last=""
while :; do
    out=$($PY "$ROOT/scripts/early_cancel_nan.py" "$JOBS" --cancel 2>&1)
    if [ "$out" != "$last" ]; then
        echo "$(date +%T)"; echo "$out"; last=$out
    fi
    states=$(sacct -j "$IDS" -X -n -o State 2>/dev/null | tr -s ' ')
    if [ -n "$states" ] && [ "$(echo "$states" | grep -vcE "$TERMINAL")" = 0 ]; then
        echo "$(date +%T) A chains drained; nan loop done"
        exit 0
    fi
    sleep 300
done
