#!/bin/bash
# Cancel a chain chunk stranded on a slow node once its last reachable
# checkpoint is safe.
#   bash scripts/cancel_stranded.sh JOBID LOG CKPT TARGET NEXT [MARGIN_S]
# Waits until LOG shows "ckpt @ TARGET" and both CKPT and the chain guard's
# CKPT.bak load at step TARGET. Then it projects the time to step NEXT from
# the median of the last five throughput readings (524,288 tokens per step)
# plus MARGIN_S (default 300). If NEXT cannot be reached before the job's
# end time, it cancels JOBID: everything computed after TARGET would be re-run
# by the next chunk anyway. The chain guard then verifies the checkpoint and
# releases the next chunk. Pick TARGET as the last checkpoint reachable at
# the current rate, counted from the current step. Used for 15496940 (c0814)
# and 15521655 (c0811) on 2026-10-16.
set -u
[ $# -ge 5 ] || { sed -n 2,13p "$0"; exit 2; }
J=$1; LOG=$2; CKPT=$3; TARGET=$4; NEXT=$5; MARGIN=${6:-300}
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
step_of() { $PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['step'])" "$1" 2>/dev/null; }
running() { sacct -j $J -X -n -o State 2>/dev/null | grep -q RUNNING; }
until grep -q "ckpt @ $TARGET" $LOG; do
    running || { echo "$(date +%T) $J no longer running; nothing to do"; exit 0; }
    sleep 60
done
echo "$(date +%T) ckpt @ $TARGET logged; waiting for the guard's .bak"
until [ "$(step_of $CKPT.bak)" = "$TARGET" ] && [ "$(step_of $CKPT)" = "$TARGET" ]; do
    running || { echo "$(date +%T) $J ended first; nothing to do"; exit 0; }
    sleep 60
done
END=$(date -d "$(squeue -j $J -h -o %e)" +%s)
S=$(grep -E "^step" $LOG | tail -n 1 | awk '{split($2,a,"/"); print a[1]}')
TOK=$(grep -E "^step" $LOG | tail -n 5 | sed -E 's/.*\(([0-9]+)k tok\/s\).*/\1/' | sort -n | sed -n 3p)
NEED=$(awk -v s=$S -v n=$NEXT -v t=$TOK -v m=$MARGIN 'BEGIN{print int((n-s)*524288/(t*1000)+m)}')
NOW=$(date +%s)
echo "$(date +%T) .bak and ckpt at step $TARGET; at step $S, median ${TOK}k tok/s: $NEXT needs ${NEED}s (incl ${MARGIN}s margin), $((END-NOW))s left"
if [ $((NOW + NEED)) -gt $END ]; then
    scancel $J && echo "$(date +%T) cancelled $J: step $NEXT unreachable before the time limit"
else
    echo "$(date +%T) step $NEXT reachable; leaving $J running"
fi
