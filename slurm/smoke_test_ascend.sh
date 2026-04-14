#!/bin/bash
#SBATCH --job-name=stj-smoke-ascend
#SBATCH --partition=debug-nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=results/smoke-ascend-%j.out
#SBATCH --error=results/smoke-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

echo "=== host ==="; hostname
echo "=== nvidia-smi ==="; nvidia-smi -L
echo "=== torch ==="
python - <<'PY'
import torch, triton
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("triton", triton.__version__)
PY
echo "=== tiny triton JIT compile check ==="
python - <<'PY'
import torch, triton, triton.language as tl
@triton.jit
def add_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(x_ptr + off, mask=m); y = tl.load(y_ptr + off, mask=m)
    tl.store(o_ptr + off, x + y, mask=m)
n = 1024
x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
o = torch.empty_like(x)
add_k[(4,)](x, y, o, n, BLOCK=256)
torch.cuda.synchronize()
assert torch.allclose(o, x + y), "triton kernel mismatch"
print("triton JIT ok on", torch.cuda.get_device_name(0))
PY
echo "=== done ==="
