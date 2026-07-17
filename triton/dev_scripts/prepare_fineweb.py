"""GPT-2 project M0: tokenize FineWeb-Edu sample-10BT into uint16 shards.

GPT-2 BPE (vocab 50257), one EOT (50256) appended per document; shards of
SHARD_TOKENS tokens written as raw uint16 .bin (nanoGPT-style) plus a
meta.json. Output + HF cache live on scratch.
"""
import json
import os

import numpy as np

SCRATCH = os.environ.get("FW_DIR", "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt")
os.makedirs(SCRATCH, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(SCRATCH, "hf_cache"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SHARD_TOKENS = 250_000_000          # 0.5 GB per shard (uint16)
NUM_PROC = max(1, (os.cpu_count() or 8) - 1)
EOT = 50256


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", num_proc=NUM_PROC)
    print(f"docs: {len(ds)}", flush=True)

    def encode(batch):
        out = tok(batch["text"], add_special_tokens=False)["input_ids"]
        return {"ids": [ids + [EOT] for ids in out],
                "len": [len(ids) + 1 for ids in out]}

    ds = ds.map(encode, batched=True, batch_size=256, num_proc=NUM_PROC,
                remove_columns=ds.column_names, desc="tokenize")

    shard = np.empty(SHARD_TOKENS, dtype=np.uint16)
    filled, shard_idx, total = 0, 0, 0
    for ex in ds:
        ids = np.asarray(ex["ids"], dtype=np.uint16)
        pos = 0
        while pos < len(ids):
            take = min(len(ids) - pos, SHARD_TOKENS - filled)
            shard[filled:filled + take] = ids[pos:pos + take]
            filled += take
            pos += take
            if filled == SHARD_TOKENS:
                path = os.path.join(SCRATCH, f"shard_{shard_idx:04d}.bin")
                shard.tofile(path)
                total += SHARD_TOKENS
                print(f"wrote {path} ({total/1e9:.2f}B tokens)", flush=True)
                shard_idx += 1
                filled = 0
    if filled:
        path = os.path.join(SCRATCH, f"shard_{shard_idx:04d}.bin")
        shard[:filled].tofile(path)
        total += filled
        print(f"wrote {path} (final, {total/1e9:.2f}B tokens)", flush=True)

    with open(os.path.join(SCRATCH, "meta.json"), "w") as f:
        json.dump({"tokenizer": "gpt2", "vocab_size": 50257, "eot": EOT,
                   "total_tokens": int(total), "shard_tokens": SHARD_TOKENS,
                   "dtype": "uint16"}, f)
    print(f"DONE: {total/1e9:.2f}B tokens in {shard_idx + 1} shards")


if __name__ == "__main__":
    main()
