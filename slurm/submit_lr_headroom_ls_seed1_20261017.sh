#!/bin/bash
# LR-headroom seed-1 replicate of the softmax-side boundary (LS@3e-4,
# LS@6e-4: softmax with A's learnable capped scale, no norm; 124M, NoPE,
# 5B tokens), pre-registered in
# thesis/experiments/2026-10-17-lr-headroom-ls-seed1-124m.md. A versioned
# copy of submit_lr_headroom_seed1_20261016.sh with the two LS cells of the
# sweep (flags as in submit_lr_headroom_2h_20261016.sh) and the sweep's
# softmax chunk count (4): same batch script, trainer, flags, --seed 1 and
# chunking (4 h with --time-min 2:00:00, c0818 excluded, --ckpt-every 250,
# --replay-rng). Submit only after the 355M seed-1 Stieltjes final
# checkpoint is verified (see the record).
# Prints "<name> <chunk1> ... <chunkN>" per run, for guard_chain.sh.
#   bash slurm/submit_lr_headroom_ls_seed1_20261017.sh LS@3e-4-s1 LS@6e-4-s1
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
[ $# -ge 1 ] || { echo "usage: $0 RUN [RUN ...]  (LS@3e-4-s1 LS@6e-4-s1)" >&2; exit 1; }
for run in "$@"; do
    case $run in
        LS@3e-4-s1) chain "$run" 4 sdpa --scale-learnable --lr 3e-4 --tag lrsweepls ;;
        LS@6e-4-s1) chain "$run" 4 sdpa --scale-learnable --lr 6e-4 --tag lrsweepls ;;
        *) echo "unknown run: $run" >&2; exit 1 ;;
    esac
done
