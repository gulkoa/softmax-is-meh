#!/bin/bash
#SBATCH --job-name=eval-sanity-q4-ref-vs-triton-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/eval-sanity-q4-ref-vs-triton-ascend-%j.out
#SBATCH --error=results/eval-sanity-q4-ref-vs-triton-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

CKPT="results/needle_stieltjes_q4.0_seq2048_nope_ascend/checkpoint.pt"

echo "=== q=4 eval at seq=2048 WITHOUT --use-triton (PyTorch ref) ==="
python nanogpt/eval_accuracy.py \
    --checkpoint "${CKPT}" \
    --task needle --attn stieltjes --q 4.0 \
    --eval-seq-len 2048 --max-arr-len 2040 \
    --val-samples 500 --batch-size 4 \
    --out "results/needle_stieltjes_q4.0_seq2048_nope_ascend/accuracy_fixed_sanity_ref.json"

echo
echo "=== q=4 eval at seq=2048 WITH --use-triton (Triton fwd) ==="
python nanogpt/eval_accuracy.py \
    --checkpoint "${CKPT}" \
    --task needle --attn stieltjes --q 4.0 \
    --eval-seq-len 2048 --max-arr-len 2040 \
    --val-samples 500 --batch-size 4 \
    --use-triton \
    --out "results/needle_stieltjes_q4.0_seq2048_nope_ascend/accuracy_fixed_sanity_triton.json"
