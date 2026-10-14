"""Re-tokenize the GPT-2 uint16 shards with a new tokenizer — offline, keeping
document order and shard indices, so <src>/shard_XXXX.bin becomes
<src>_<tag>/shard_XXXX.bin and the LAST shard of each source is still the
trainer's validation shard (the same documents). One EOT (id 0) per document
exactly where the GPT-2 stream had one (a document cut by a shard boundary
gets none at the cut). meta.json: tokenizer name + sha256, vocab, eot.

  python retokenize_shards.py --tokenizer <dir with tokenizer.json> --tag tok32k
         [--sources web,math,code] [--workers 24] [--threads 4] [--limit-tokens N]
"""
import argparse, hashlib, json, os, time
from multiprocessing import Pool
import numpy as np

B = "/fs/scratch/PAS2836/alexg"
SOURCES = {"web": f"{B}/fineweb_edu_10bt", "math": f"{B}/finemath_4plus", "code": f"{B}/codeparrot_py"}
GPT2_EOT = 50256
CHUNK = 50_000_000
_G = {}


def _init(tok_dir, threads):
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    _G["gpt2"] = AutoTokenizer.from_pretrained("gpt2")
    _G["new"] = Tokenizer.from_file(f"{tok_dir}/tokenizer.json")
    _G["eot"] = _G["new"].token_to_id("<|endoftext|>")


def one_shard(job):
    src_path, dst_path, limit = job
    gpt2, new, eot = _G["gpt2"], _G["new"], _G["eot"]
    m = np.memmap(src_path, dtype=np.uint16, mode="r")
    n = len(m) if not limit else min(len(m), limit)
    carry = np.empty(0, dtype=np.int64)
    pos, docs_n, tok_in, tok_out, rt_ok, rt_n = 0, 0, 0, 0, 0, 0
    t0 = time.time()
    with open(dst_path + ".tmp", "wb") as f:
        def emit(pieces, terminated):
            nonlocal docs_n, tok_out, rt_ok, rt_n
            texts = gpt2.batch_decode(pieces, skip_special_tokens=True)
            encs = new.encode_batch(texts)
            out = []
            for text, e, term in zip(texts, encs, terminated):
                ids = e.ids + ([eot] if term else [])
                out.extend(ids)
                if rt_n < 20:
                    rt_n += 1; rt_ok += (new.decode(e.ids) == text)
            docs_n += sum(terminated); tok_out += len(out)
            np.asarray(out, dtype=np.uint16).tofile(f)
        while pos < n:
            chunk = np.asarray(m[pos:pos + min(CHUNK, n - pos)]).astype(np.int64)
            pos += len(chunk)
            buf = np.concatenate([carry, chunk]) if len(carry) else chunk
            cuts = np.flatnonzero(buf == GPT2_EOT)
            pieces, start = [], 0
            for c in cuts:
                pieces.append(buf[start:c].tolist()); start = c + 1
            carry = buf[start:]
            tok_in += len(chunk)
            if pieces:
                emit(pieces, [True] * len(pieces))
        if len(carry):                       # document continues in the next shard: no EOT
            emit([carry.tolist()], [False])
    os.replace(dst_path + ".tmp", dst_path)
    return dict(src=src_path, dst=dst_path, gpt2_tokens=tok_in, new_tokens=tok_out, docs=docs_n,
                roundtrip=f"{rt_ok}/{rt_n}", seconds=round(time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tag", default="tok32k")
    ap.add_argument("--sources", default="web,math,code")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit-tokens", type=int, default=0, help="debug: only the first N GPT-2 tokens of each shard")
    ap.add_argument("--out-base", default=B)
    a = ap.parse_args()
    man = json.load(open(f"{a.tokenizer}/manifest.json")) if os.path.exists(f"{a.tokenizer}/manifest.json") else {}
    sha = hashlib.sha256(open(f"{a.tokenizer}/tokenizer.json", "rb").read()).hexdigest()
    jobs, dsts = [], {}
    for src in a.sources.split(","):
        d = SOURCES[src]
        out = f"{a.out_base}/{os.path.basename(d)}_{a.tag}"
        os.makedirs(out, exist_ok=True)
        dsts[src] = out
        for sh in sorted(f for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".bin")):
            jobs.append((f"{d}/{sh}", f"{out}/{sh}", a.limit_tokens))
    print(f"{len(jobs)} shards → {a.workers} workers × {a.threads} threads", flush=True)
    stats = []
    with Pool(a.workers, initializer=_init, initargs=(a.tokenizer, a.threads)) as pool:
        for r in pool.imap_unordered(one_shard, jobs):
            stats.append(r)
            print(f"  {os.path.basename(os.path.dirname(r['dst']))}/{os.path.basename(r['dst'])}: "
                  f"{r['gpt2_tokens']/1e6:.0f}M → {r['new_tokens']/1e6:.0f}M tokens, {r['docs']} docs, "
                  f"rt {r['roundtrip']}, {r['seconds']}s", flush=True)
    for src, out in dsts.items():
        ss = sorted((s for s in stats if s["dst"].startswith(out + "/")), key=lambda s: s["dst"])
        meta = {"tokenizer": os.path.basename(a.tokenizer.rstrip("/")), "tokenizer_path": a.tokenizer,
                "tokenizer_sha256": sha, "vocab_size": man.get("vocab_size", 32768), "eot": 0, "dtype": "uint16",
                "source_dir": SOURCES[src], "shards": len(ss),
                "total_tokens": int(sum(s["new_tokens"] for s in ss)),
                "gpt2_total_tokens": int(sum(s["gpt2_tokens"] for s in ss)),
                "per_shard_tokens": [int(s["new_tokens"]) for s in ss], "docs": int(sum(s["docs"] for s in ss))}
        json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
        print(f"{src}: {meta['gpt2_total_tokens']/1e9:.2f}B GPT-2 tokens → {meta['total_tokens']/1e9:.2f}B "
              f"({meta['total_tokens']/max(1, meta['gpt2_total_tokens']):.3f}×), {meta['shards']} shards, meta.json written", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
