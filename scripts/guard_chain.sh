#!/bin/bash
# Guard every boundary of a linear afterany chain of chunks that share one
# in-place checkpoint. For each consecutive pair: wait until the earlier chunk
# runs, hold the next one, then scripts/guard_boundary.sh (backups while it
# runs; after it ends, verify the checkpoint, restore the .bak if it does not
# load, release the next chunk). If the guard dies, a chunk can be left in
# JobHeldUser: verify the checkpoint loads, then scontrol release it.
#
#   bash scripts/guard_chain.sh CKPT JOB1 JOB2 ... JOBn
set -u
CKPT=$1; shift
JOBS=("$@")
ROOT=/users/PAS2402/alexg/softmax/softmax-is-meh
TERMINAL="COMPLETED|TIMEOUT|FAILED|CANCELLED|NODE_FAIL|OUT_OF_ME|PREEMPTED|DEADLINE"
for ((i = 0; i + 1 < ${#JOBS[@]}; i++)); do
    J=${JOBS[i]}; N=${JOBS[i+1]}
    until sacct -j "$J" -X -n -o State 2>/dev/null | grep -qE "RUNNING|$TERMINAL"; do
        sleep 120
    done
    # never verify (or restore) under a chunk that is already writing the checkpoint
    if ! scontrol hold "$N" 2>/dev/null; then
        echo "$(date +%T) $N could not be held ($(sacct -j "$N" -X -n -o State | tr -s ' ')); boundary $J -> $N unguarded"
        continue
    fi
    echo "$(date +%T) $J running; holding $N"
    bash $ROOT/scripts/guard_boundary.sh "$J" "$N" "$CKPT" || exit 1
done
echo "$(date +%T) chain guarded to its end"
