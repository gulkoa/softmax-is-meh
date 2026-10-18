#!/bin/bash
# Seed-1 replicate of the 124M modern pair E (Stieltjes) / F (softmax),
# pre-registered in thesis/experiments/2026-10-18-ef-seed1-124m.md. A
# versioned copy of submit_lr_headroom_ls_seed1_20261017.sh for seed 0's
# batch script, slurm/backbone_pilot_h100.sh (unchanged: backbone worktree
# trainer, NoPE + QK-norm + modern, cap 0, learnable scale, lr 1e-3, 5B
# tokens). Seed 0's flags plus --seed 1 and --ckpt-every 250; F adds
# --attn sdpa as seed 0 did. Chunks 4 h with --time-min 2:00:00, afterany,
# c0806, c0807 and c0818 excluded; E 6 chunks, F 5 (worst case 2 h each,
# plus a spare).
# Prints "<name> <chunk1> ... <chunkN>" per run, for guard_chain.sh.
#   bash slurm/submit_ef_seed1_20261018.sh E-s1 F-s1
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/backbone_pilot_h100.sh
COMMON="--seed 1 --ckpt-every 250"
X=c0806,c0807,c0818
chain() {
    local name=$1 n=$2; shift 2
    local ids j prev
    prev=$(sbatch --parsable -t 4:00:00 --time-min 2:00:00 -x $X "$S" "$@" $COMMON)
    ids=$prev
    for _ in $(seq 2 "$n"); do
        j=$(sbatch --parsable -t 4:00:00 --time-min 2:00:00 -x $X --dependency=afterany:$prev "$S" "$@" $COMMON)
        ids="$ids $j"
        prev=$j
    done
    echo "$name $ids"
}
[ $# -ge 1 ] || { echo "usage: $0 RUN [RUN ...]  (E-s1 F-s1)" >&2; exit 1; }
for run in "$@"; do
    case $run in
        E-s1) chain "$run" 6 ;;
        F-s1) chain "$run" 5 --attn sdpa ;;
        *) echo "unknown run: $run" >&2; exit 1 ;;
    esac
done
