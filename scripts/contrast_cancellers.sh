#!/bin/bash
# Keep one spare canceller (scripts/cancel_spares_at_final.sh) alive per
# 1e-3 contrast chain until the chain has drained. Written 2026-10-27 to
# replace the session-scratchpad loop lost with that temp directory. That
# loop ran the canceller once per chain; on 10-27 16:10 a Slurm controller
# timeout made squeue return nothing, the canceller took the chains for
# drained and exited while all nine A jobs were still pending. Here a chain
# counts as drained only when sacct (the accounting database, which kept
# answering during that timeout) shows every job of the chain in a
# terminal state; on any other exit the canceller is restarted after 120 s.
# Prints the canceller's lines prefixed with the chain label.
#   bash scripts/contrast_cancellers.sh        # the three A chains (LS done 10-21)
set -u
ROOT=/users/PAS2402/alexg/softmax/softmax-is-meh
FINAL=${FINAL:-9500}
TERMINAL="COMPLETED|TIMEOUT|FAILED|CANCELLED|NODE_FAIL|OUT_OF_ME|PREEMPTED|DEADLINE"
chain() {
    local label=$1; shift
    local ids; ids=$(echo "$@" | tr ' ' ',')
    while :; do
        local states; states=$(sacct -j "$ids" -X -n -o State 2>/dev/null | tr -s ' ')
        if [ -z "$states" ]; then sleep 120; continue; fi
        if [ "$(echo "$states" | grep -vcE "$TERMINAL")" = 0 ]; then
            echo "[$label] $(date +%T) chain drained ($(echo "$states" | tr '\n' ' '| tr -s ' '))"
            return 0
        fi
        bash "$ROOT/scripts/cancel_spares_at_final.sh" "$FINAL" "$@" 2>&1 | sed "s/^/[$label] /"
        sleep 120
    done
}
chain A@1e-3-s2 15669636 15669637 15669638 15669639 15669640 &
chain A@1e-3-s3 15669641 15669642 15669643 15669644 15669645 &
chain A@1e-3-s4 15669646 15669647 15669648 15669649 15669650 &
wait
echo "$(date +%T) all chains drained"
