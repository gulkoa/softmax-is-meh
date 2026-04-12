"""
CLRS task data generation for nanoGPT training.

Vocabulary:
  0-255: integer values (array elements, node IDs)
  256:   SEPARATOR token
  257:   PAD token
  VOCAB_SIZE = 258
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------
SEPARATOR = 256
PAD = 257
VOCAB_SIZE = 258


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    task_name: str
    seq_len: int = 128       # total sequence length
    max_arr_len: int = 16    # max input array length
    max_val: int = 64        # max integer value
    num_samples: int = 10000


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _encode_sequence(input_arr: List[int], output_arr: List[int], seq_len: int) -> List[int]:
    """Concatenate input + [SEPARATOR] + output, pad to seq_len, truncate if over."""
    tokens = input_arr + [SEPARATOR] + output_arr
    if len(tokens) < seq_len:
        tokens = tokens + [PAD] * (seq_len - len(tokens))
    else:
        tokens = tokens[:seq_len]
    return tokens


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------
def _generate_sorting(cfg: TaskConfig) -> List[int]:
    """Random array -> sorted array."""
    arr_len = random.randint(4, cfg.max_arr_len)
    arr = [random.randint(0, cfg.max_val - 1) for _ in range(arr_len)]
    sorted_arr = sorted(arr)
    return _encode_sequence(arr, sorted_arr, cfg.seq_len)


def _generate_binary_search(cfg: TaskConfig) -> List[int]:
    """Sorted unique array + target -> index of target."""
    arr_len = random.randint(4, cfg.max_arr_len)
    # Sample unique values and sort them
    population = list(range(0, cfg.max_val))
    arr = sorted(random.sample(population, min(arr_len, len(population))))
    target_idx = random.randint(0, len(arr) - 1)
    target = arr[target_idx]
    input_part = arr + [SEPARATOR, target]
    output_part = [target_idx]
    return _encode_sequence(input_part, output_part, cfg.seq_len)


def _generate_bfs(cfg: TaskConfig) -> List[int]:
    """Random connected graph -> BFS visit order from node 0."""
    n_nodes = random.randint(4, min(cfg.max_arr_len, 256))

    # Build a random spanning tree to ensure connectivity
    adj = [[] for _ in range(n_nodes)]
    for i in range(1, n_nodes):
        parent = random.randint(0, i - 1)
        adj[i].append(parent)
        adj[parent].append(i)

    # Add a few extra random edges
    extra_edges = random.randint(0, n_nodes // 2)
    for _ in range(extra_edges):
        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)
        if u != v and v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)

    # Sort adjacency lists
    for i in range(n_nodes):
        adj[i].sort()

    # BFS from node 0
    visited = []
    seen = set()
    queue = deque([0])
    seen.add(0)
    while queue:
        node = queue.popleft()
        visited.append(node)
        for neighbor in adj[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    # Encode: [n_nodes, len(adj[0]), adj[0]..., len(adj[1]), adj[1]..., ..., SEPARATOR, visit_order..., PAD, ...]
    input_part = [n_nodes]
    for i in range(n_nodes):
        input_part.append(len(adj[i]))
        input_part.extend(adj[i])

    return _encode_sequence(input_part, visited, cfg.seq_len)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
def _generate_max(cfg: TaskConfig) -> List[int]:
    """Random array -> [max_value, max_index]."""
    arr_len = random.randint(4, cfg.max_arr_len)
    arr = [random.randint(0, cfg.max_val - 1) for _ in range(arr_len)]
    max_val = max(arr)
    max_idx = arr.index(max_val)
    return _encode_sequence(arr, [max_val, max_idx], cfg.seq_len)


TASK_GENERATORS = {
    "sorting": _generate_sorting,
    "binary_search": _generate_binary_search,
    "bfs": _generate_bfs,
    "max": _generate_max,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CLRSDataset(Dataset):
    def __init__(self, cfg: TaskConfig, seed: int = 42):
        random.seed(seed)
        gen = TASK_GENERATORS[cfg.task_name]
        self.samples = [gen(cfg) for _ in range(cfg.num_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def test_data():
    for task_name in TASK_GENERATORS:
        cfg = TaskConfig(task_name=task_name, num_samples=100)
        ds = CLRSDataset(cfg)
        x, y = ds[0]
        print(f"  [{task_name}] x.shape={x.shape}, y.shape={y.shape}, x[:10]={x[:10].tolist()}")
        assert x.shape == y.shape == (cfg.seq_len - 1,)
    print("All data tests passed.")


if __name__ == "__main__":
    test_data()
