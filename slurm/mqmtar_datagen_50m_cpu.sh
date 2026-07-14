#!/bin/bash
#SBATCH --job-name=mqmtar-datagen-50m
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-datagen-50m-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-datagen-50m-%j.err

# Full published MQMTAR training set: their README command verbatim
# (train_size 50,000,000 = 390,625 steps x batch 128, one epoch).
# ~2.9k lines/s observed for the 3.2M set => ~5h. Eval splits regenerate
# identically (same generator defaults) but the trainer only reads test_*
# from this dir too, keeping everything self-contained.

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  asentmax/scripts/generate_data.py \
    --task_type mqmtar \
    --out_dir softmax-is-meh/results/mqmtar_data/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4 \
    --train_size 50000000 \
    --vocab_size 10000 \
    --seq_len 48 \
    --vary_len 16 \
    --mdps_seq_len "64 128 256 512 1024 2048 4096 8192 16384 32768 65536" \
    --mdps_vary_len "0 0 0 0 0 0 0 0 0 0 0" \
    --abc_size 256 \
    --k_len 2 \
    --v_len 2 \
    --num_q 4 \
    --num_kv 0.8

echo "DATAGEN DONE"
ls -la softmax-is-meh/results/mqmtar_data/
