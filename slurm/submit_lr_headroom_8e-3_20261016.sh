#!/bin/bash
# Conditional stage of the LR-headroom sweep, pre-registered in
# thesis/experiments/2026-10-15-lr-headroom-124m.md: A@8e-3 if A is clean
# at 4e-3, plus D@8e-3 if D is also clean at 4e-3 (the ceiling: nothing
# above 8e-3). Submit only when scripts/score_lr_headroom.py prints
# "CONDITIONAL STAGE TRIGGERED". Same protocol and chunking as
# submit_lr_headroom_20261015.sh (two 4 h chunks, --replay-rng, c0818
# excluded). Prints "<name> <chunk1> <chunk2>" per run: append the lines to
# results/lr_headroom_jobs.txt and guard each chain with guard_chain.sh.
#   bash slurm/submit_lr_headroom_8e-3_20261016.sh A [D]
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/qknorm_pilot_h100.sh
COMMON="--seed 0 --probe-every 20 --replay-rng"
sub() {
    local name=$1; shift
    local j1 j2
    j1=$(sbatch --parsable -t 4:00:00 -x c0818 "$S" "$@" $COMMON)
    j2=$(sbatch --parsable -t 4:00:00 -x c0818 --dependency=afterany:$j1 "$S" "$@" $COMMON)
    echo "$name $j1 $j2"
}
[ $# -ge 1 ] || { echo "usage: $0 A [D]" >&2; exit 1; }
for arm in "$@"; do
    case $arm in
        A) sub A@8e-3 stj  --scale-learnable --lr 8e-3 --tag lrsweep ;;
        D) sub D@8e-3 sdpa --qk-norm         --lr 8e-3 --tag lrsweep ;;
        *) echo "unknown arm: $arm (A or D)" >&2; exit 1 ;;
    esac
done
