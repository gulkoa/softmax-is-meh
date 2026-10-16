#!/bin/bash
#SBATCH --job-name=kspeed-skip
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-causal-skip-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-causal-skip-%j.err
# Causal tile skip in the forward and delta kernels (branch kernel-speed,
# worktree .worktrees/kernel-speed; record
# thesis/experiments/2026-10-16-kernel-causal-skip.md). Runs, on the branch:
# 1. the test suite on a GPU (its GPU section included), via a copy that
#    points at the worktree and the main venv (the worktree has no .venv);
# 2. the kernel's self-tests (forward/backward vs the PyTorch reference,
#    non-contiguous layouts, long-N backward);
# 3. bit-identity against the pre-change kernel (dc8dd3d) + timings;
# 4. the step-time split with the branch kernel at the 355M and 124M shapes.
# Reads main's trainer read-only; writes only results/.
set -uo pipefail
W=/users/PAS2402/alexg/softmax/.worktrees/kernel-speed
R=/users/PAS2402/alexg/softmax/softmax-is-meh/results
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
T=$(mktemp -d)
sed -e "s|^REPO_DIR=.*|REPO_DIR=$W|" \
    -e "s|^source .*activate\"\$|source /users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/activate|" \
    $W/scripts/test_all.sh > $T/test_all.sh
grep -n '^REPO_DIR\|^source' $T/test_all.sh
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== 1. test suite (worktree, GPU)"
bash $T/test_all.sh; echo "suite exit $?"
echo "== 2. kernel self-tests"
cd $W/triton
$PY - <<'PYEOF'
import sys
sys.path.insert(0, "/users/PAS2402/alexg/softmax/.worktrees/kernel-speed/triton")
import stieltjes_flash_attn as m
assert m.__file__.startswith("/users/PAS2402/alexg/softmax/.worktrees/kernel-speed/"), m.__file__
print("kernel:", m.__file__)
ok = [m.test_forward_correctness(), m.test_backward_correctness(), m.test_noncontiguous_layout()]
m.test_long_n_backward_nan()
print("self-tests fwd/bwd/noncontig:", ok, "PASS" if all(ok) else "FAIL")
PYEOF
echo "self-tests exit $?"
echo "== 3. bit-identity vs dc8dd3d + timings"
$PY $W/triton/test_causal_skip.py --bench; echo "identity exit $?"
echo "== 4. step split with the branch kernel"
$PY $W/triton/dev_scripts/profile_step_split.py --kernel $W/triton/stieltjes_flash_attn.py \
    --layers 24 --heads 16 --embd 1024 --micro-bs 8 --grad-accum 64 \
    --json $R/profile_step_split_355m_causalskip_${SLURM_JOB_ID}.json
$PY $W/triton/dev_scripts/profile_step_split.py --kernel $W/triton/stieltjes_flash_attn.py \
    --layers 12 --heads 12 --embd 768 --micro-bs 16 --grad-accum 32 \
    --json $R/profile_step_split_124m_causalskip_${SLURM_JOB_ID}.json
echo "ALL DONE"
