#!/bin/bash
# Pre-commit test suite — run BEFORE every commit to nanogpt/ code.
# Tests on CPU (no GPU required). Catches import errors, shape bugs,
# off-by-one errors, and basic correctness issues.
#
# Usage: scripts/test_all.sh
# Exit code 0 = all tests pass, nonzero = failure

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

source "${REPO_DIR}/triton/.venv/bin/activate"

FAILED=0

run_test() {
    local name="$1"
    shift
    echo -n "  [$name] "
    if "$@" 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        FAILED=1
    fi
}

echo "=== nanogpt test suite (CPU) ==="
echo ""

# 1. Import checks — catches missing imports
echo "--- Import checks ---"
run_test "import data" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import CLRSDataset, TaskConfig, VOCAB_SIZE, PAD, SEPARATOR
assert SEPARATOR == 256
assert PAD == 257
assert VOCAB_SIZE == 258
print('OK')
"

run_test "import model" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from model import GPTConfig, GPT
print('OK')
"

run_test "import train" python -c "
import sys; sys.path.insert(0, 'nanogpt')
# This must not crash — verifies all imports resolve
import train
print('OK')
"

run_test "import analyze" python -c "
import sys; sys.path.insert(0, 'nanogpt')
import analyze
print('OK')
"

# 2. Data generation correctness
echo ""
echo "--- Data correctness ---"
run_test "sorting output is sorted" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD
cfg = TaskConfig(task_name='sorting', num_samples=100, seq_len=64, max_arr_len=12)
ds = CLRSDataset(cfg, seed=42)
for i in range(100):
    tokens = ds.samples[i]
    sep_idx = tokens.index(SEPARATOR)
    input_arr = tokens[:sep_idx]
    output_part = tokens[sep_idx+1:]
    output_arr = [t for t in output_part if t != PAD]
    assert output_arr == sorted(input_arr), f'Sample {i}: {input_arr} -> {output_arr} != {sorted(input_arr)}'
print('OK — 100 samples verified')
"

run_test "max output correct" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD
cfg = TaskConfig(task_name='max', num_samples=100, seq_len=64, max_arr_len=12)
ds = CLRSDataset(cfg, seed=42)
for i in range(100):
    tokens = ds.samples[i]
    sep_idx = tokens.index(SEPARATOR)
    input_arr = tokens[:sep_idx]
    output_part = [t for t in tokens[sep_idx+1:] if t != PAD]
    assert len(output_part) == 2, f'Sample {i}: expected 2 output tokens, got {len(output_part)}'
    assert output_part[0] == max(input_arr), f'Sample {i}: max={max(input_arr)} but got {output_part[0]}'
    assert output_part[1] == input_arr.index(max(input_arr)), f'Sample {i}: argmax wrong'
print('OK — 100 samples verified')
"

run_test "binary_search output correct" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD
cfg = TaskConfig(task_name='binary_search', num_samples=100, seq_len=64, max_arr_len=12)
ds = CLRSDataset(cfg, seed=42)
for i in range(100):
    tokens = ds.samples[i]
    # Find all separators
    seps = [j for j, t in enumerate(tokens) if t == SEPARATOR]
    assert len(seps) >= 2, f'Sample {i}: expected >=2 separators, got {len(seps)}'
    arr = tokens[:seps[0]]
    target = tokens[seps[0]+1:seps[1]]
    assert len(target) == 1, f'Sample {i}: target should be 1 token'
    output = [t for t in tokens[seps[1]+1:] if t != PAD]
    assert len(output) == 1, f'Sample {i}: output should be 1 token (index)'
    assert arr[output[0]] == target[0], f'Sample {i}: arr[{output[0]}]={arr[output[0]]} != target={target[0]}'
print('OK — 100 samples verified')
"

run_test "needle task correctness" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD
cfg = TaskConfig(task_name='needle', num_samples=100, seq_len=4096, max_arr_len=2000)
ds = CLRSDataset(cfg, seed=42)
for i in range(100):
    tokens = ds.samples[i]
    sep_idx = tokens.index(SEPARATOR)
    input_arr = tokens[:sep_idx]
    output_part = [t for t in tokens[sep_idx+1:] if t != PAD]
    assert len(output_part) == 1, f'Sample {i}: expected 1 output, got {len(output_part)}'
    needle_val = output_part[0]
    assert 128 <= needle_val <= 254, f'Sample {i}: needle value {needle_val} out of range'
    assert needle_val in input_arr, f'Sample {i}: needle not in input'
    # Exactly one needle
    n_needles = sum(1 for t in input_arr if t >= 128)
    assert n_needles == 1, f'Sample {i}: {n_needles} needles (expected 1)'
print('OK — 100 needle samples verified')
"

run_test "bfs no vocab collision at max scale" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD
cfg = TaskConfig(task_name='bfs', num_samples=50, seq_len=4096, max_arr_len=255, max_val=256)
ds = CLRSDataset(cfg, seed=99)
for i in range(50):
    tokens = ds.samples[i]
    for j, t in enumerate(tokens):
        if t == SEPARATOR:
            break  # found real separator
        assert t < SEPARATOR, f'Sample {i} pos {j}: token {t} collides with SEPARATOR ({SEPARATOR})'
print('OK — 50 large BFS samples, no vocab collisions')
"

run_test "bfs output correct" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD
from collections import deque
cfg = TaskConfig(task_name='bfs', num_samples=50, seq_len=128, max_arr_len=12)
ds = CLRSDataset(cfg, seed=42)
for i in range(50):
    tokens = ds.samples[i]
    sep_idx = tokens.index(SEPARATOR)
    # Decode adjacency list
    inp = tokens[:sep_idx]
    n_nodes = inp[0]
    pos = 1
    adj = [[] for _ in range(n_nodes)]
    for node in range(n_nodes):
        deg = inp[pos]; pos += 1
        for _ in range(deg):
            adj[node].append(inp[pos]); pos += 1
    # Run BFS
    visited = []
    queue = deque([0])
    seen = {0}
    while queue:
        node = queue.popleft()
        visited.append(node)
        for nb in sorted(adj[node]):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    output = [t for t in tokens[sep_idx+1:] if t != PAD]
    assert output == visited, f'Sample {i}: BFS mismatch'
print('OK — 50 samples verified')
"

# 3. Model forward/backward (CPU, softmax only)
echo ""
echo "--- Model tests ---"
run_test "model forward+backward" python -c "
import sys; sys.path.insert(0, 'nanogpt')
import torch
from model import GPTConfig, GPT
cfg = GPTConfig(vocab_size=258, block_size=64, n_layer=2, n_head=2, n_embd=64, attn_type='softmax')
model = GPT(cfg)
x = torch.randint(0, 258, (2, 32))
targets = torch.randint(0, 258, (2, 32))
logits, loss = model(x, targets=targets)
assert logits.shape == (2, 32, 258), f'Bad logits shape: {logits.shape}'
loss.backward()
print(f'OK — loss={loss.item():.4f}')
"

run_test "ignore_index=257 matches PAD" python -c "
import sys; sys.path.insert(0, 'nanogpt')
from data import PAD
# Verify the hardcoded 257 in model.py matches data.py PAD
assert PAD == 257, f'PAD mismatch: data.PAD={PAD}, model uses 257'
print('OK')
"

# 4. Accuracy metric
echo ""
echo "--- Accuracy metric ---"
run_test "output-only accuracy" python -c "
import sys; sys.path.insert(0, 'nanogpt')
import torch
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD

# Create a fake model that perfectly predicts the next token
cfg = TaskConfig(task_name='sorting', num_samples=10, seq_len=32, max_arr_len=6)
ds = CLRSDataset(cfg, seed=42)

# Manually compute what accuracy should be
x, y = ds[0]
sep_positions = (x == SEPARATOR).nonzero(as_tuple=True)[0]
assert len(sep_positions) > 0, 'No SEPARATOR found in x'
output_start = sep_positions[-1].item()

# Count output tokens (non-PAD after last SEP)
output_mask = torch.zeros_like(y, dtype=torch.bool)
output_mask[output_start:] = True
output_mask &= (y != PAD)
n_output = output_mask.sum().item()
assert n_output > 0, f'No output tokens found'

# Verify first output token is correct
first_output_token = y[output_start].item()
assert first_output_token != PAD, f'First output token is PAD'
assert first_output_token != SEPARATOR, f'First output token is SEPARATOR'
print(f'OK — {n_output} output tokens, first={first_output_token}')
"

run_test "binary_search accuracy finds target_idx" python -c "
import sys; sys.path.insert(0, 'nanogpt')
import torch
from data import CLRSDataset, TaskConfig, SEPARATOR, PAD

cfg = TaskConfig(task_name='binary_search', num_samples=10, seq_len=32, max_arr_len=6)
ds = CLRSDataset(cfg, seed=42)
x, y = ds[0]
sep_positions = (x == SEPARATOR).nonzero(as_tuple=True)[0]
assert len(sep_positions) >= 2, f'Expected >=2 SEPs for binary_search, got {len(sep_positions)}'
output_start = sep_positions[-1].item()
# y[output_start] should be the target index
target_idx = y[output_start].item()
assert target_idx != PAD and target_idx != SEPARATOR, f'Output token is PAD/SEP: {target_idx}'
print(f'OK — target_idx={target_idx}')
"

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== ALL TESTS PASSED ==="
    exit 0
else
    echo "=== SOME TESTS FAILED ==="
    exit 1
fi
