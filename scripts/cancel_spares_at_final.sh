#!/bin/bash
# Cancel a sweep run's spare chunks before they can start, once the run is
# certain to finish inside its current chunk.
#   bash scripts/cancel_spares_at_final.sh FINAL_VAL_STEP JOB1 JOB2 ... JOBn
# Why: the sweep trainer (qknorm worktree) re-saves the final checkpoint in
# place, not atomically, when a chunk starts after FINAL. A spare must
# therefore never be cancelled while it runs, because a cancel during that
# save can corrupt the finished run's only final checkpoint. (10-16 17:35:
# two spares started 9 s after release and were cancelled 25 s in, before
# their save.) Polls every 60 s. When a RUNNING chunk's log shows
# "[val] step FINAL_VAL_STEP" with at least 10 min left before its time
# limit, it scancels every later chunk that is still PENDING (held or
# waiting on a dependency, so none has started). If that line appears with
# less than 10 min left, or the chain ends first, it cancels nothing. The
# chain guard's release of a cancelled chunk then fails harmlessly.
# DRY=1 prints what it would cancel instead.
set -u
[ $# -ge 2 ] || { sed -n 2,17p "$0"; exit 2; }
FINAL=$1; shift
JOBS=("$@")
LOGDIR=/users/PAS2402/alexg/softmax/softmax-is-meh/results
while :; do
    alive=0
    for ((i = 0; i < ${#JOBS[@]}; i++)); do
        J=${JOBS[i]}
        st=$(squeue -j "$J" -h -o %T 2>/dev/null)
        [ -n "$st" ] && alive=1
        [ "$st" = RUNNING ] || continue
        grep -q "\[val\] step $FINAL " "$LOGDIR/qknorm-pilot-$J.out" 2>/dev/null || continue
        left=$(( $(date -d "$(squeue -j "$J" -h -o %e)" +%s) - $(date +%s) ))
        if [ "$left" -lt 600 ]; then
            echo "$(date +%T) $J logged step $FINAL with ${left}s left; cancelling nothing"
            exit 0
        fi
        later=()
        for ((k = i + 1; k < ${#JOBS[@]}; k++)); do
            [ "$(squeue -j "${JOBS[k]}" -h -o %T 2>/dev/null)" = PENDING ] && later+=("${JOBS[k]}")
        done
        if [ ${#later[@]} -gt 0 ]; then
            if [ -n "${DRY:-}" ]; then
                echo "$(date +%T) DRY: $J logged step $FINAL with ${left}s left; would cancel pending ${later[*]}"
            else
                scancel "${later[@]}" && echo "$(date +%T) $J logged step $FINAL with ${left}s left; cancelled pending ${later[*]}"
            fi
        else
            echo "$(date +%T) $J logged step $FINAL; no pending chunk after it"
        fi
        exit 0
    done
    [ "$alive" = 1 ] || { echo "$(date +%T) no chunk of $* queued or running; nothing to do"; exit 0; }
    sleep 60
done
