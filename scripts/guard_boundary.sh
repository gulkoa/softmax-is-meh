#!/bin/bash
# Guard one chain boundary of a trainer that saves in place with no SIGTERM
# handling. While JOB runs, keep a verified <CKPT>.bak (scripts/backup_ckpts.py).
# Once sacct reports JOB in a terminal state, check that CKPT loads; if it does
# not (killed mid-save), move it aside to <CKPT>.corrupt and restore the .bak.
# Then release SUCCESSOR, which the caller must have held (scontrol hold) so it
# cannot start on a corrupt checkpoint. Login node; run in the background.
#
#   scontrol hold SUCCESSOR
#   bash scripts/guard_boundary.sh JOB SUCCESSOR CKPT
set -u
JOB=$1; SUCC=$2; CKPT=$3
ROOT=/users/PAS2402/alexg/softmax/softmax-is-meh
PY=$ROOT/triton/.venv/bin/python
TERMINAL="COMPLETED|TIMEOUT|FAILED|CANCELLED|NODE_FAIL|OUT_OF_ME|PREEMPTED|DEADLINE"
step_of() {
    $PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['step'])" "$1" 2>/dev/null
}
# a transient sacct failure prints nothing and keeps the loop waiting
until sacct -j "$JOB" -X -n -o State 2>/dev/null | grep -qE "$TERMINAL"; do
    $PY $ROOT/scripts/backup_ckpts.py "$CKPT" --settle 90 2>&1 | grep -v "backup current"
    sleep 60
done
sleep 20
echo "$(date +%T) $JOB ended: $(sacct -j "$JOB" -X -n -o State,Elapsed | tr -s ' ')"
S=$(step_of "$CKPT")
if [ -n "$S" ]; then
    echo "ckpt loads, step $S"
else
    echo "ckpt FAILED to load; restoring $CKPT.bak"
    mv "$CKPT" "$CKPT.corrupt"
    cp "$CKPT.bak" "$CKPT.tmp" && mv "$CKPT.tmp" "$CKPT"
    S=$(step_of "$CKPT")
    if [ -z "$S" ]; then
        echo "restored ckpt does not load either; $SUCC stays held"
        exit 1
    fi
    echo "restored step $S"
fi
scontrol release "$SUCC" && echo "$(date +%T) released $SUCC"
