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
    needle_margin: str = "distinctive"  # "distinctive" (default, +128 above bg) or "subtle" (+1 above bg)


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
    # Cap at 255 so n_nodes token (encoded as first input token) stays in 0-255 vocab range
    # n_nodes=256 would collide with SEPARATOR token (256)
    n_nodes = random.randint(4, min(cfg.max_arr_len, 255))

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
def _generate_needle(cfg: TaskConfig) -> List[int]:
    """Needle-in-haystack: find the one 'needle' token in a sea of 'background' tokens.

    Two modes (cfg.needle_margin):
      - "distinctive" (default): background ∈ [0,127], needle ∈ [128,254].
        Targets are categorically distinguishable, so token-value matching alone
        suffices regardless of context length — softmax does not dilute.
      - "subtle": background ∈ [0,127] but the needle is uniquely-valued in
        the haystack and is the *maximum* of the array. Output: needle value.
        With margin=1, the needle's logit only marginally exceeds distractors,
        which is the regime where softmax dilution actually bites at long
        context.
    """
    arr_len = random.randint(cfg.max_arr_len // 2, cfg.max_arr_len)
    if cfg.needle_margin == "subtle":
        # Sample a per-sample ceiling so the needle value spans a real range.
        # The earlier implementation drew background uniformly from [0, 126]
        # and set needle = max(background)+1; at the arr_len we use, max
        # saturates near 126 with high probability, making needle near-constant
        # and letting a "always output 127" baseline score near-perfectly. A
        # per-sample cap decouples the needle value from the background cap
        # while preserving the +1 margin property.
        cap = random.randint(20, 126)
        arr = [random.randint(0, cap - 1) for _ in range(arr_len)]
        needle_pos = random.randint(0, arr_len - 1)
        needle_val = cap  # strictly greater than every other token by >= 1
        arr[needle_pos] = needle_val
    else:
        # Default distinctive mode (current behaviour).
        arr = [random.randint(0, 127) for _ in range(arr_len)]
        needle_pos = random.randint(0, arr_len - 1)
        needle_val = random.randint(128, 254)
        arr[needle_pos] = needle_val
    return _encode_sequence(arr, [needle_val], cfg.seq_len)


def _generate_max(cfg: TaskConfig) -> List[int]:
    """Random array -> [max_value, max_index] (or just [max_value] for long arrays).

    The index is dropped from the output when max_arr_len > 256 because
    positions > 255 cannot be represented in the current vocab. This unblocks
    the long-context dilution test (where attention over thousands of input
    tokens is the point of the experiment).
    """
    if cfg.max_arr_len > 256:
        # Long-context mode: arr can span the full configured length; output
        # is just the max value. Min is 256 so we still get a meaningful
        # haystack even on the short end of the random range.
        arr_len = random.randint(256, cfg.max_arr_len)
        arr = [random.randint(0, cfg.max_val - 1) for _ in range(arr_len)]
        max_val = max(arr)
        return _encode_sequence(arr, [max_val], cfg.seq_len)
    # Default short-context mode (unchanged): arr capped at 256, output
    # includes max_index.
    arr_len = random.randint(4, min(cfg.max_arr_len, 256))
    arr = [random.randint(0, cfg.max_val - 1) for _ in range(arr_len)]
    max_val = max(arr)
    max_idx = arr.index(max_val)
    return _encode_sequence(arr, [max_val, max_idx], cfg.seq_len)


def _generate_top2_needle(cfg: TaskConfig) -> List[int]:
    """Top-2 needle: subtle margin, random gap. Forces multi-modal attention.

    - top1_val ∈ [20, 126] (random per sample): the array max.
    - top2_val = top1_val − delta where delta ∈ {1, 2, 3} (random per sample):
      subtle margin (≤ 3) but 3 possibilities so model can't just predict
      "top1 − 1" without actually attending to the top-2 position.
    - Background ∈ [0, top2_val − 1] strictly: all other elements below top-2.
    - Output: [top1_val, top2_val]. Model must attend to BOTH planted positions
      (single-mode attention to top-1 alone does not give top-2).
    """
    arr_len = random.randint(cfg.max_arr_len // 2, cfg.max_arr_len)
    top1_val = random.randint(20, 126)
    delta = random.randint(1, 3)
    top2_val = max(top1_val - delta, 1)  # avoid 0 so background still has range
    arr = [random.randint(0, top2_val - 1) for _ in range(arr_len)]
    pos1, pos2 = random.sample(range(arr_len), 2)
    arr[pos1] = top1_val
    arr[pos2] = top2_val
    return _encode_sequence(arr, [top1_val, top2_val], cfg.seq_len)


TASK_GENERATORS = {
    "sorting": _generate_sorting,
    "binary_search": _generate_binary_search,
    "bfs": _generate_bfs,
    "max": _generate_max,
    "needle": _generate_needle,
    "top2_needle": _generate_top2_needle,
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
