#!/bin/bash
# LR-headroom seed-1 replicate of the two non-monotone Stieltjes cells
# (A@2e-3, A@4e-3; 124M, NoPE, 5B tokens), pre-registered in
# thesis/experiments/2026-10-16-lr-headroom-a-seed1-124m.md. A versioned
# copy of submit_lr_headroom_2h_20261016.sh with --seed 1 and only these
# two cells: same batch script, trainer, flags and chunking (4 h with
# --time-min 2:00:00, c0818 excluded, --ckpt-every 250, --replay-rng).
# Prints "<name> <chunk1> ... <chunkN>" per run, for guard_chain.sh.
#   bash slurm/submit_lr_headroom_seed1_20261016.sh A@2e-3-s1 A@4e-3-s1
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/qknorm_pilot_h100.sh
COMMON="--seed 1 --probe-every 20 --replay-rng --ckpt-every 250"
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
[ $# -ge 1 ] || { echo "usage: $0 RUN [RUN ...]  (A@2e-3-s1 A@4e-3-s1)" >&2; exit 1; }
for run in "$@"; do
    case $run in
        A@2e-3-s1) chain "$run" 5 stj --scale-learnable --lr 2e-3 --tag lrsweep ;;
        A@4e-3-s1) chain "$run" 5 stj --scale-learnable --lr 4e-3 --tag lrsweep ;;
        *) echo "unknown run: $run" >&2; exit 1 ;;
    esac
done
