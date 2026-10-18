#!/bin/bash
# The 124M no-norm contrast at lr 1e-3, seeds 2, 3 and 4: A (Stieltjes with
# the learnable capped scale) and LS (softmax with A's learnable capped
# scale), no norm; 124M, NoPE, 5B tokens. Pre-registered in
# thesis/experiments/2026-10-18-1e-3-contrast-seeds-124m.md. A versioned copy
# of submit_a_2e3_seeds_20261018.sh: same batch script, trainer, flags and
# chunking style (4 h chunks with --time-min 2:00:00, afterany,
# --ckpt-every 250, --replay-rng, --probe-every 20); A's arguments as there,
# LS's as in submit_ls_3e4_seeds_20261018.sh; five chunks per A run, four per
# LS run; the two exclusions standing at submission (c0806, c0818); every
# chunk carries --nice=20000 so the A@2e-3 seed chunks stay ahead.
# Prints "<name> <chunk1> ... <chunkN>" per run, for guard_chain.sh.
#   bash slurm/submit_1e3_contrast_seeds_20261018.sh A@1e-3-s2 A@1e-3-s3 \
#       A@1e-3-s4 LS@1e-3-s2 LS@1e-3-s3 LS@1e-3-s4
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/qknorm_pilot_h100.sh
X=c0806,c0818
chain() {
    local name=$1 n=$2 seed=$3; shift 3
    local ids j prev common="--seed $seed --probe-every 20 --replay-rng --ckpt-every 250"
    prev=$(sbatch --parsable --nice=20000 -t 4:00:00 --time-min 2:00:00 -x $X "$S" "$@" $common)
    ids=$prev
    for _ in $(seq 2 "$n"); do
        j=$(sbatch --parsable --nice=20000 -t 4:00:00 --time-min 2:00:00 -x $X --dependency=afterany:$prev "$S" "$@" $common)
        ids="$ids $j"
        prev=$j
    done
    echo "$name $ids"
}
[ $# -ge 1 ] || { echo "usage: $0 RUN [RUN ...]  (A@1e-3-s2..s4 LS@1e-3-s2..s4)" >&2; exit 1; }
for run in "$@"; do
    case $run in
        A@1e-3-s2) chain "$run" 5 2 stj --scale-learnable --lr 1e-3 --tag lrsweep ;;
        A@1e-3-s3) chain "$run" 5 3 stj --scale-learnable --lr 1e-3 --tag lrsweep ;;
        A@1e-3-s4) chain "$run" 5 4 stj --scale-learnable --lr 1e-3 --tag lrsweep ;;
        LS@1e-3-s2) chain "$run" 4 2 sdpa --scale-learnable --lr 1e-3 --tag lrsweepls ;;
        LS@1e-3-s3) chain "$run" 4 3 sdpa --scale-learnable --lr 1e-3 --tag lrsweepls ;;
        LS@1e-3-s4) chain "$run" 4 4 sdpa --scale-learnable --lr 1e-3 --tag lrsweepls ;;
        *) echo "unknown run: $run" >&2; exit 1 ;;
    esac
done
