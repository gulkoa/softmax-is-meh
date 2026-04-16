#!/bin/bash
# Submitter (run from login node): 1-layer max scaling sweep on Ascend.
# 4 stj q ∈ {8,16,32,64} × 2 seqs + 2 softmax seqs = 10 jobs.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh

for Q in 8.0 16.0 32.0 64.0; do
    for SEQ_BS in "2048 8" "4096 4"; do
        SEQ=$(echo $SEQ_BS | cut -d' ' -f1)
        BS=$(echo $SEQ_BS | cut -d' ' -f2)
        sbatch -J "n1-stj-q${Q}-s${SEQ}-ascend" \
            --export=ALL,ATTN=stieltjes,Q=${Q},SEQ=${SEQ},BS=${BS} \
            slurm/train_max_1layer_scaling_template_ascend.sh
    done
done

for SEQ_BS in "2048 8" "4096 4"; do
    SEQ=$(echo $SEQ_BS | cut -d' ' -f1)
    BS=$(echo $SEQ_BS | cut -d' ' -f2)
    sbatch -J "n1-sm-s${SEQ}-ascend" \
        --export=ALL,ATTN=softmax,Q=1.0,SEQ=${SEQ},BS=${BS} \
        slurm/train_max_1layer_scaling_template_ascend.sh
done
