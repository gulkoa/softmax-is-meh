#!/bin/bash
# (Re)start the login-node helpers of the three held A@1e-3 contrast chains
# after a login-node reboot (11-05 killed all of them). Starts, detached
# from the shell with their logs under /fs/scratch/PAS2836/alexg/helper_logs:
# the three chain guards (scripts/guard_chain.sh), the spare cancellers
# (scripts/contrast_cancellers.sh), the nan loop (scripts/contrast_nan_loop.sh)
# and the snapshot keepers (scripts/contrast_snapshots.sh). Refuses to start
# anything while any helper is still running, so it never doubles them.
# Expected process count afterwards (the tick check's pgrep pattern): 15.
#   bash scripts/restart_contrast_helpers.sh
set -u
ROOT=/users/PAS2402/alexg/softmax/softmax-is-meh
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
LOGS=/fs/scratch/PAS2836/alexg/helper_logs
PAT='^bash .*(guard_chain.sh|contrast_cancellers.sh|contrast_snapshots.sh|contrast_nan_loop.sh|cancel_spares_at_final.sh)'
n=$(pgrep -fc "$PAT")
if [ "$n" != 0 ]; then
    echo "$n helper processes already running; not starting any:"
    pgrep -fa "$PAT" | sed 's/^/  /'
    exit 1
fi
mkdir -p "$LOGS"
stamp=$(date +%F_%H%M%S)
start() {
    local name=$1; shift
    nohup setsid bash "$@" > "$LOGS/$name-$stamp.log" 2>&1 < /dev/null &
    echo "$(date +%T) started $name (pid $!) -> $LOGS/$name-$stamp.log"
}
start guard-s2 "$ROOT/scripts/guard_chain.sh" "$D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.pt" 15669638 15669639 15669640
start guard-s3 "$ROOT/scripts/guard_chain.sh" "$D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s3.pt" 15669643 15669644 15669645
start guard-s4 "$ROOT/scripts/guard_chain.sh" "$D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s4.pt" 15669648 15669649 15669650
start cancellers "$ROOT/scripts/contrast_cancellers.sh"
start nan-loop "$ROOT/scripts/contrast_nan_loop.sh"
start snapshots "$ROOT/scripts/contrast_snapshots.sh"
sleep 5
echo "$(date +%T) helper processes now: $(pgrep -fc "$PAT") (expected 15)"
