#!/bin/bash
# LR-headroom sweep (124M, NoPE, 5B tokens, seed 0) in short chunks,
# pre-registered in thesis/experiments/2026-10-15-lr-headroom-124m.md
# (commit f026130; re-chunking amendment 2026-10-16). Supersedes the 4 h
# chunking of submit_lr_headroom_20261015.sh (cancelled unstarted) and of
# submit_lr_headroom_8e-3_20261016.sh: 2 h jobs backfill far sooner here.
# Each chunk asks for 4 h with --time-min 2:00:00, so the scheduler may
# shorten it to as little as 2 h to start it in an earlier gap.
# Same trainer, flags and seed; --ckpt-every 250 halves the steps re-run
# after each time limit; --replay-rng makes every continuation draw the
# batches of an uninterrupted run. Chunks: Stieltjes (A, ~7.1 h of
# training) 5, softmax (D, LS, ~4.8 h) 4; a chunk that starts after its run
# finished re-saves the final checkpoint and exits in minutes.
# Prints "<name> <chunk1> ... <chunkN>" per run, for guard_chain.sh.
#   bash slurm/submit_lr_headroom_2h_20261016.sh A@2e-3 A@4e-3 D@2e-3 ...
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/qknorm_pilot_h100.sh
COMMON="--seed 0 --probe-every 20 --replay-rng --ckpt-every 250"
chain() {
    local name=$1 n=$2; shift 2
    local ids j prev
    prev=$(sbatch --parsable -t 4:00:00 --time-min 2:00:00 -x c0818 "$S" "$@" $COMMON)
    ids=$prev
    for _ in $(seq 2 "$n"); do
        j=$(sbatch --parsable -t 4:00:00 --time-min 2:00:00 -x c0818 --dependency=afterany:$prev "$S" "$@" $COMMON)
        ids="$ids $j"
        prev=$j
    done
    echo "$name $ids"
}
[ $# -ge 1 ] || { echo "usage: $0 RUN [RUN ...]  (A@2e-3 A@4e-3 A@8e-3 D@2e-3 D@4e-3 D@8e-3 LS@3e-4 LS@6e-4)" >&2; exit 1; }
for run in "$@"; do
    case $run in
        A@2e-3)  chain "$run" 5 stj  --scale-learnable --lr 2e-3 --tag lrsweep ;;
        A@4e-3)  chain "$run" 5 stj  --scale-learnable --lr 4e-3 --tag lrsweep ;;
        A@8e-3)  chain "$run" 5 stj  --scale-learnable --lr 8e-3 --tag lrsweep ;;
        D@2e-3)  chain "$run" 4 sdpa --qk-norm         --lr 2e-3 --tag lrsweep ;;
        D@4e-3)  chain "$run" 4 sdpa --qk-norm         --lr 4e-3 --tag lrsweep ;;
        D@8e-3)  chain "$run" 4 sdpa --qk-norm         --lr 8e-3 --tag lrsweep ;;
        LS@3e-4) chain "$run" 4 sdpa --scale-learnable --lr 3e-4 --tag lrsweepls ;;
        LS@6e-4) chain "$run" 4 sdpa --scale-learnable --lr 6e-4 --tag lrsweepls ;;
        *) echo "unknown run: $run" >&2; exit 1 ;;
    esac
done
