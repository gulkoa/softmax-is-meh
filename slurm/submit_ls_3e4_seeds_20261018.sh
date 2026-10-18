#!/bin/bash
# LS@3e-4 at seeds 2, 3 and 4 (softmax with A's learnable capped scale, no
# norm; 124M, NoPE, 5B tokens), pre-registered in
# thesis/experiments/2026-10-18-ls-3e-4-seeds-124m.md. A versioned copy of
# submit_lr_headroom_ls_seed1_20261017.sh: same batch script, trainer, flags
# and chunking (4 chunks of 4 h with --time-min 2:00:00, afterany,
# --ckpt-every 250, --replay-rng, --probe-every 20); the seed differs, the
# exclusions are the three standing ones (c0806, c0807, c0818) and every
# chunk carries --nice=100000 so the E/F seed-1 chunks stay ahead.
# Prints "<name> <chunk1> ... <chunkN>" per run, for guard_chain.sh.
#   bash slurm/submit_ls_3e4_seeds_20261018.sh LS@3e-4-s2 LS@3e-4-s3 LS@3e-4-s4
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/qknorm_pilot_h100.sh
X=c0806,c0807,c0818
chain() {
    local name=$1 n=$2 seed=$3; shift 3
    local ids j prev common="--seed $seed --probe-every 20 --replay-rng --ckpt-every 250"
    prev=$(sbatch --parsable --nice=100000 -t 4:00:00 --time-min 2:00:00 -x $X "$S" "$@" $common)
    ids=$prev
    for _ in $(seq 2 "$n"); do
        j=$(sbatch --parsable --nice=100000 -t 4:00:00 --time-min 2:00:00 -x $X --dependency=afterany:$prev "$S" "$@" $common)
        ids="$ids $j"
        prev=$j
    done
    echo "$name $ids"
}
[ $# -ge 1 ] || { echo "usage: $0 RUN [RUN ...]  (LS@3e-4-s2 LS@3e-4-s3 LS@3e-4-s4)" >&2; exit 1; }
for run in "$@"; do
    case $run in
        LS@3e-4-s2) chain "$run" 4 2 sdpa --scale-learnable --lr 3e-4 --tag lrsweepls ;;
        LS@3e-4-s3) chain "$run" 4 3 sdpa --scale-learnable --lr 3e-4 --tag lrsweepls ;;
        LS@3e-4-s4) chain "$run" 4 4 sdpa --scale-learnable --lr 3e-4 --tag lrsweepls ;;
        *) echo "unknown run: $run" >&2; exit 1 ;;
    esac
done
