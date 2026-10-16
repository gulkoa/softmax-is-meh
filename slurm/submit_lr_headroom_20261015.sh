#!/bin/bash
# LR-headroom sweep (124M, NoPE, 5B tokens, seed 0), pre-registered in
# thesis/experiments/2026-10-15-lr-headroom-124m.md (commit f026130).
# Two 4 h chunks per run, the second resumed with --replay-rng (afterany);
# node c0818 excluded (ran the probe run and the smoke at a fraction of
# normal speed). Prints "<name> <chunk1> <chunk2>" per run, for guard_chain.sh.
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
sub A@2e-3  stj  --scale-learnable --lr 2e-3 --tag lrsweep
sub A@4e-3  stj  --scale-learnable --lr 4e-3 --tag lrsweep
sub D@2e-3  sdpa --qk-norm         --lr 2e-3 --tag lrsweep
sub D@4e-3  sdpa --qk-norm         --lr 4e-3 --tag lrsweep
sub LS@3e-4 sdpa --scale-learnable --lr 3e-4 --tag lrsweepls
sub LS@6e-4 sdpa --scale-learnable --lr 6e-4 --tag lrsweepls
