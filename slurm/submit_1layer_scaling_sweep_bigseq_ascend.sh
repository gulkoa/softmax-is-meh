#!/bin/bash
# 1-layer max scaling sweep: seq ∈ {8192, 16384} on Ascend.
# 4 stj q × 2 seqs + 2 softmax × 2 seqs = 10 jobs
# bs=2 at 8k, bs=1 at 16k
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh

for Q in 8.0 16.0 32.0 64.0; do
    for SEQ_BS in "8192 2" "16384 1"; do
        SEQ=$(echo $SEQ_BS | cut -d' ' -f1)
        BS=$(echo $SEQ_BS | cut -d' ' -f2)
        sbatch -J "n1-stj-q${Q}-s${SEQ}-ascend" \
            --time=04:00:00 \
            --export=ALL,ATTN=stieltjes,Q=${Q},SEQ=${SEQ},BS=${BS} \
            slurm/train_max_1layer_scaling_template_ascend.sh
    done
done

for SEQ_BS in "8192 2" "16384 1"; do
    SEQ=$(echo $SEQ_BS | cut -d' ' -f1)
    BS=$(echo $SEQ_BS | cut -d' ' -f2)
    sbatch -J "n1-sm-s${SEQ}-ascend" \
        --time=04:00:00 \
        --export=ALL,ATTN=softmax,Q=1.0,SEQ=${SEQ},BS=${BS} \
        slurm/train_max_1layer_scaling_template_ascend.sh
done
