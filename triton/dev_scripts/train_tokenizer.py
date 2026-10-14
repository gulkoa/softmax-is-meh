"""Train the Stilt 32k tokenizer (roadmap item 6; design
thesis/findings/2026-10-14-plan-tokenizer-32k.md).

Byte-level BPE (HF `tokenizers`) with the cl100k-style pre-tokenizer,
16 reserved special ids, trained on whole documents decoded from the
existing GPT-2 shards in the 70/15/15 proportion (TRAIN shards only — the
last shard of each source is the trainer's validation shard and is never
read here). Offline: no Hub access needed beyond the cached GPT-2 tokenizer.

Writes <out>/tokenizer.json, the PreTrainedTokenizerFast files
(tokenizer_config.json, special_tokens_map.json) and manifest.json
(sample composition, sha256, held-out bytes/token vs GPT-2, round-trip).

  python train_tokenizer.py --out /fs/scratch/PAS2836/alexg/tokenizers/stilt-tok-32k-v1
"""
import argparse, hashlib, json, os, time
import numpy as np
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex

B = "/fs/scratch/PAS2836/alexg"
SOURCES = {"web": f"{B}/fineweb_edu_10bt", "math": f"{B}/finemath_4plus", "code": f"{B}/codeparrot_py"}
GPT2_EOT = 50256
CL100K_PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIAL = (["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|eot|>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
           + [f"<|reserved_{i}|>" for i in range(9, 16)])
CHUNK = 50_000_000


def iter_docs(path, tok, from_end=False, limit=None):
    """Yield (text, n_gpt2_tokens) for whole documents of one shard, in order.
    A document cut by the shard boundary is yielded as its piece."""
    m = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(m)
    if from_end and limit:
        m = m[-limit:]; n = len(m)
    carry = np.empty(0, dtype=np.int64)
    pos = 0
    while pos < n:
        chunk = np.asarray(m[pos:pos + CHUNK]).astype(np.int64)
        pos += len(chunk)
        buf = np.concatenate([carry, chunk]) if len(carry) else chunk
        cuts = np.flatnonzero(buf == GPT2_EOT)
        start = 0
        docs = []
        for c in cuts:
            if c > start:
                docs.append(buf[start:c].tolist())
            start = c + 1
        carry = buf[start:]
        if docs:
            for ids, text in zip(docs, tok.batch_decode(docs, skip_special_tokens=True)):
                yield text, len(ids)
    if len(carry):
        yield tok.decode(carry.tolist(), skip_special_tokens=True), len(carry)


def collect(src, d, tok, budget, manifest):
    shards = sorted(f for f in os.listdir(d) if f.startswith("shard_"))[:-1]   # never the val shard
    texts, used, got = [], [], 0
    for sh in shards:
        used.append(sh)
        for text, n in iter_docs(f"{d}/{sh}", tok):
            texts.append(text); got += n
            if got >= budget:
                break
        if got >= budget:
            break
    manifest["sample"][src] = {"shards": used, "docs": len(texts), "gpt2_tokens": got,
                               "bytes": sum(len(t.encode("utf-8")) for t in texts)}
    print(f"{src}: {len(texts)} docs, {got/1e6:.0f}M GPT-2 tokens from {used}", flush=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--web-tokens", type=float, default=700e6)
    ap.add_argument("--math-tokens", type=float, default=150e6)
    ap.add_argument("--code-tokens", type=float, default=150e6)
    ap.add_argument("--eval-tokens", type=float, default=4e6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    manifest = {"name": os.path.basename(a.out.rstrip("/")), "vocab_size": a.vocab, "special_tokens": SPECIAL,
                "pre_tokenizer": "cl100k regex + ByteLevel", "regex": CL100K_PAT, "sample": {}, "seed": "deterministic shard order"}
    t0 = time.time()
    texts = []
    for src, budget in (("web", a.web_tokens), ("math", a.math_tokens), ("code", a.code_tokens)):
        texts += collect(src, SOURCES[src], gpt2, int(budget), manifest)
    print(f"sample: {len(texts)} docs, {sum(len(t) for t in texts)/1e9:.2f}G chars, decoded in {time.time()-t0:.0f}s", flush=True)
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(CL100K_PAT), behavior="isolated"),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=SPECIAL, show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t0 = time.time()
    tk.train_from_iterator(texts, trainer=tr, length=len(texts))
    manifest["train_seconds"] = round(time.time() - t0)
    assert tk.get_vocab_size() == a.vocab, tk.get_vocab_size()
    assert [tk.token_to_id(s) for s in SPECIAL] == list(range(len(SPECIAL))), "special ids must be 0..15"
    tk.save(f"{a.out}/tokenizer.json")
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                                   pad_token="<|pad|>", additional_special_tokens=SPECIAL[2:],
                                   model_max_length=1000000000000)
    fast.save_pretrained(a.out)
    manifest["sha256_tokenizer_json"] = hashlib.sha256(open(f"{a.out}/tokenizer.json", "rb").read()).hexdigest()
    # held-out check on the val shards: bytes/token vs GPT-2 + exact round trip
    manifest["heldout"] = {}
    for src, d in SOURCES.items():
        last = sorted(f for f in os.listdir(d) if f.startswith("shard_"))[-1]
        docs = [t for t, _ in iter_docs(f"{d}/{last}", gpt2, from_end=True, limit=int(a.eval_tokens))]
        nb = sum(len(t.encode("utf-8")) for t in docs)
        n_new = sum(len(e.ids) for e in tk.encode_batch(docs))
        n_old = sum(len(x) for x in gpt2(docs, add_special_tokens=False)["input_ids"])
        rt = sum(tk.decode(tk.encode(t).ids) == t for t in docs[:200])
        manifest["heldout"][src] = {"val_shard": last, "docs": len(docs), "bytes": nb, "bytes_per_token": nb / n_new,
                                    "gpt2_bytes_per_token": nb / n_old, "tokens_vs_gpt2": n_new / n_old,
                                    "roundtrip_exact": f"{rt}/{min(200, len(docs))}"}
        print(f"held-out {src}: {nb/n_new:.3f} B/tok (GPT-2 {nb/n_old:.3f}) → {n_new/n_old:.3f}× tokens; round-trip {rt}/{min(200, len(docs))}", flush=True)
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
