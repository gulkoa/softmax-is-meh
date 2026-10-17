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
# "[val] step FINAL_VAL_STEP" with at least MIN_LEFT (default 10 min) left
# before its time limit, it scancels every later chunk that is still PENDING
# (held or waiting on a dependency, so none has started). If that line
# appears with less left, or the chain ends first, it cancels nothing. The
# chain guard's release of a cancelled chunk then fails harmlessly.
# DRY=1 prints what it would cancel instead. PREFIX selects the chunk logs
# (default qknorm-pilot, i.e. results/qknorm-pilot-<job>.out; the 355M
# chain's are PREFIX=nope-qk-stj). The 355M main trainer returns early
# ("FINAL already reached") instead of re-saving, so there a spare only
# wastes its start-up and delays jobs that wait on the chain's tail.
# MIN_LEFT (seconds, default 600) is the time the chunk must have left when
# the final validation appears; it must cover the steps after that
# validation plus the final save (124M: 36 steps, ~2 min; 355M: 109 steps
# + a 4.2 GB save, ~15 min, so use MIN_LEFT=1500 there).
set -u
[ $# -ge 2 ] || { sed -n 2,26p "$0"; exit 2; }
MIN_LEFT=${MIN_LEFT:-600}
FINAL=$1; shift
JOBS=("$@")
LOGDIR=/users/PAS2402/alexg/softmax/softmax-is-meh/results
PREFIX=${PREFIX:-qknorm-pilot}
while :; do
    alive=0
    for ((i = 0; i < ${#JOBS[@]}; i++)); do
        J=${JOBS[i]}
        st=$(squeue -j "$J" -h -o %T 2>/dev/null)
        [ -n "$st" ] && alive=1
        [ "$st" = RUNNING ] || continue
        grep -q "\[val\] step $FINAL " "$LOGDIR/$PREFIX-$J.out" 2>/dev/null || continue
        left=$(( $(date -d "$(squeue -j "$J" -h -o %e)" +%s) - $(date +%s) ))
        if [ "$left" -lt "$MIN_LEFT" ]; then
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
