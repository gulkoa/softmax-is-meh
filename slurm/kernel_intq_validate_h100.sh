#!/bin/bash
#SBATCH --job-name=kernel-intq-val
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-intq-val-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-intq-val-%j.err

# Validate + re-bench the integer-q fast path (_inv_pow_pair: reciprocal +
# multiply chains replacing log/exp for q in {1,2,3,4,8,16}; constexpr,
# resolves at compile time). Gate: built-in fwd/bwd self-tests AND the
# normalized suite (vs Jack's reference) must pass before the perf matrix.

set -euo pipefail
cd /users/PAS2402/alexg/softmax

echo "########## 1. built-in self-tests + benchmark ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/stieltjes_flash_attn.py

echo ""
echo "########## 2. normalized-mode suite (vs Jack's reference) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/test_triton_normalized.py

echo ""
echo "########## 3. BH-scaling perf matrix (int-q chains) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/bench_flashn_bh_scaling.py

echo "ALL DONE"
