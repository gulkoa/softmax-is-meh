#!/bin/bash
#SBATCH --job-name=kspeed-main
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-main-gpu-suite-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-main-gpu-suite-%j.err
# GPU suite on MAIN after the kernel-speed merge (merge plan step 2 of
# thesis/experiments/2026-10-16-kernel-causal-skip.md; record
# thesis/experiments/2026-10-18-kernel-speed-merge.md). Versioned from
# kernel_speed_causal_skip_debug.sh and kernel_speed_real_identity_debug.sh,
# pointed at main instead of the worktree. Runs:
# 1. scripts/test_all.sh with its GPU section;
# 2. the kernel's self-tests (forward/backward vs the PyTorch reference,
#    non-contiguous layouts, long-N backward);
# 3. bit-identity against the pre-change kernel (dc8dd3d) + timings;
# 4. one real micro-step through the old and merged kernels on the two
#    high-score sweep snapshots (loss and every gradient bit for bit).
# Reads checkpoints read-only; writes only results/.
set -uo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh
PY=$S/triton/.venv/bin/python
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
git -C $S log --oneline -1
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== 1. test suite (main, GPU)"
bash $S/scripts/test_all.sh; echo "suite exit $?"
echo "== 2. kernel self-tests"
cd $S/triton
$PY - <<'PYEOF'
import sys
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
import stieltjes_flash_attn as m
assert m.__file__.startswith("/users/PAS2402/alexg/softmax/softmax-is-meh/triton/"), m.__file__
print("kernel:", m.__file__)
ok = [m.test_forward_correctness(), m.test_backward_correctness(), m.test_noncontiguous_layout()]
m.test_long_n_backward_nan()
print("self-tests fwd/bwd/noncontig:", ok, "PASS" if all(ok) else "FAIL")
PYEOF
echo "self-tests exit $?"
echo "== 3. bit-identity vs dc8dd3d + timings"
$PY $S/triton/test_causal_skip.py --bench; echo "identity exit $?"
echo "== 4. real micro-step identity on high-score snapshots"
$PY $S/triton/check_real_identity.py \
    $D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s0.step4000.pt \
    $D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.004_s0.step3000.pt; echo "real exit $?"
echo "ALL DONE"
