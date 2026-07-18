"""Tokenize any HF text dataset into GPT-2-BPE uint16 shards (generic
version of prepare_fineweb.py) — for the 355M mixed-corpus run.

Usage: python prepare_hf_corpus.py --dataset HuggingFaceTB/finemath \
         --name finemath-4plus --out /fs/scratch/.../finemath_4plus
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=float, default=0,
                    help="stop after this many tokens (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(
        os.path.dirname(args.out.rstrip("/")), "hf_cache"))

    from datasets import load_dataset
    from transformers import AutoTokenizer

    SHARD_TOKENS = 250_000_000
    NUM_PROC = max(1, (os.cpu_count() or 8) - 1)
    EOT = 50256
    tok = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset(args.dataset, name=args.name, split=args.split,
                      num_proc=NUM_PROC)
    print(f"{args.dataset}/{args.name}: {len(ds)} docs", flush=True)

    key = args.text_key

    def encode(batch):
        out = tok(batch[key], add_special_tokens=False)["input_ids"]
        return {"ids": [ids + [EOT] for ids in out]}

    ds = ds.map(encode, batched=True, batch_size=256, num_proc=NUM_PROC,
                remove_columns=ds.column_names, desc="tokenize")

    shard = np.empty(SHARD_TOKENS, dtype=np.uint16)
    filled, shard_idx, total = 0, 0, 0
    limit = int(args.max_tokens) if args.max_tokens else None
    for ex in ds:
        ids = np.asarray(ex["ids"], dtype=np.uint16)
        pos = 0
        while pos < len(ids):
            take = min(len(ids) - pos, SHARD_TOKENS - filled)
            shard[filled:filled + take] = ids[pos:pos + take]
            filled += take
            pos += take
            if filled == SHARD_TOKENS:
                path = os.path.join(args.out, f"shard_{shard_idx:04d}.bin")
                shard.tofile(path)
                total += SHARD_TOKENS
                print(f"wrote {path} ({total/1e9:.2f}B tokens)", flush=True)
                shard_idx += 1
                filled = 0
                if limit and total >= limit:
                    break
        if limit and total >= limit:
            print("token limit reached", flush=True)
            break
    if filled and not (limit and total >= limit):
        path = os.path.join(args.out, f"shard_{shard_idx:04d}.bin")
        shard[:filled].tofile(path)
        total += filled
        print(f"wrote {path} (final)", flush=True)
        shard_idx += 1

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"tokenizer": "gpt2", "dataset": args.dataset,
                   "name": args.name, "total_tokens": int(total),
                   "dtype": "uint16"}, f)
    print(f"DONE: {total/1e9:.2f}B tokens in {shard_idx} shards -> {args.out}")


if __name__ == "__main__":
    main()
