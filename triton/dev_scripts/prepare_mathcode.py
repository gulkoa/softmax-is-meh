"""Task #31 data: tokenize math (FineMath-4+) and code (codeparrot-clean
Python) into GPT-2-BPE uint16 shard dirs alongside the FineWeb-Edu web
shards, for the 355M web+math+code mix."""
import json
import os
import sys

import numpy as np

BASE = "/fs/scratch/PAS2836/alexg"
os.environ.setdefault("HF_HOME", os.path.join(BASE, "fineweb_edu_10bt",
                                              "hf_cache"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SHARD_TOKENS = 250_000_000
NUM_PROC = max(1, (os.cpu_count() or 8) - 1)
EOT = 50256

SOURCES = {
    "math": dict(path="HuggingFaceTB/finemath", name="finemath-4plus",
                 text_col="text", target_tokens=4e9,
                 out=os.path.join(BASE, "finemath_4plus")),
    "code": dict(path="codeparrot/codeparrot-clean", name=None,
                 text_col="content", target_tokens=4e9,
                 out=os.path.join(BASE, "codeparrot_py")),
}


def prepare(tag):
    src = SOURCES[tag]
    os.makedirs(src["out"], exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset(src["path"], name=src["name"], split="train",
                      streaming=True)
    shard = np.empty(SHARD_TOKENS, dtype=np.uint16)
    filled, shard_idx, total = 0, 0, 0
    buf = []
    for ex in ds:
        buf.append(ex[src["text_col"]])
        if len(buf) < 256:
            continue
        for ids in tok(buf, add_special_tokens=False)["input_ids"]:
            ids = np.asarray(ids + [EOT], dtype=np.uint16)
            pos = 0
            while pos < len(ids):
                take = min(len(ids) - pos, SHARD_TOKENS - filled)
                shard[filled:filled + take] = ids[pos:pos + take]
                filled += take
                pos += take
                if filled == SHARD_TOKENS:
                    p = os.path.join(src["out"], f"shard_{shard_idx:04d}.bin")
                    shard.tofile(p)
                    total += SHARD_TOKENS
                    shard_idx += 1
                    filled = 0
                    print(f"[{tag}] {total/1e9:.2f}B tokens", flush=True)
        buf = []
        if total >= src["target_tokens"]:
            break
    if filled and total < src["target_tokens"]:
        p = os.path.join(src["out"], f"shard_{shard_idx:04d}.bin")
        shard[:filled].tofile(p)
        total += filled
        shard_idx += 1
    with open(os.path.join(src["out"], "meta.json"), "w") as f:
        json.dump({"tokenizer": "gpt2", "eot": EOT,
                   "total_tokens": int(total),
                   "shard_tokens": SHARD_TOKENS}, f)
    print(f"[{tag}] DONE: {total/1e9:.2f}B tokens, {shard_idx} shards "
          f"-> {src['out']}", flush=True)


if __name__ == "__main__":
    prepare(sys.argv[1])
