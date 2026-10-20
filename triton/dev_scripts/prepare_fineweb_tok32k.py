"""FineWeb sample-100BT -> stilt-tok-32k-v1 uint16 shards: the stilt.1.2 data
(plan amendment 2026-10-20 in thesis/findings/2026-10-14-plan-stilt12-355m-pair.md;
record thesis/experiments/2026-10-20-fineweb-100bt-tok32k.md).

  python prepare_fineweb_tok32k.py download --out <dir> --revision <sha>
         [--repo HuggingFaceFW/fineweb] [--subdir sample/100BT] [--workers 16] [--list-only]
  python prepare_fineweb_tok32k.py build --parquet <download dir> --out <new dir>
         --tokenizer <dir with tokenizer.json> [--exclude-shard <old val shard.bin>]
         [--shard-tokens 250000000] [--workers 24] [--threads 4] [--batch-rows 8192]

download: snapshot of <subdir>/*.parquet at a pinned revision; every file's size
is checked against the repository listing; manifest.json lists path, size and
LFS sha256.
build: (1) the exclude shard is split at EOT and decoded; each document is
hashed exactly and with all whitespace removed. (2) Every parquet file becomes
<out>/parts/part_XXXX.bin, one EOT after each kept document; a document whose
whitespace-stripped hash is in the exclude set is dropped. Resumable per file.
(3) The parts, in file order, are cut into K = round(T / shard_tokens) shards,
each at the first document end at or after a multiple of T/K, so every shard
ends with EOT and the LAST shard is the trainer's validation shard. Each part
is deleted once merged. Refuses an --out that already holds shards.
meta.json carries the trainer's keys and no gpt2_total_tokens (the stilt12
budget helper would read it): the launch passes --total-tokens explicitly.
"""
import argparse, glob, hashlib, json, os, time
import multiprocessing as mp
import numpy as np

CHUNK = 64_000_000
_G = {}


def _digest(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()


def _ws_digest(s):
    return _digest("".join(s.split()))


# ---------------------------------------------------------------- download

def download(a):
    from huggingface_hub import HfApi, snapshot_download
    files = [f for f in HfApi().list_repo_tree(a.repo, path_in_repo=a.subdir, repo_type="dataset",
                                               revision=a.revision)
             if getattr(f, "size", None) is not None and f.path.endswith(".parquet")]
    total = sum(f.size for f in files)
    print(f"{a.repo}@{a.revision} {a.subdir}: {len(files)} files, {total / 1e9:.1f} GB", flush=True)
    if a.list_only:
        return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    snapshot_download(repo_id=a.repo, repo_type="dataset", revision=a.revision,
                      allow_patterns=[f"{a.subdir}/*.parquet"], local_dir=a.out, max_workers=a.workers)
    bad = [f.path for f in files
           if not os.path.exists(os.path.join(a.out, f.path))
           or os.path.getsize(os.path.join(a.out, f.path)) != f.size]
    man = {"repo": a.repo, "repo_type": "dataset", "revision": a.revision, "subdir": a.subdir,
           "files": [{"path": f.path, "size": f.size,
                      "sha256": getattr(getattr(f, "lfs", None), "sha256", None)} for f in files],
           "total_bytes": total, "seconds": round(time.time() - t0)}
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    if bad:
        raise SystemExit(f"{len(bad)} files missing or the wrong size, e.g. {bad[:5]}")
    print(f"download OK: {len(files)} files, {total / 1e9:.1f} GB in {man['seconds']} s", flush=True)


# ------------------------------------------------------------------- build

def exclude_sets(shard, tok_file):
    """(exact digests, whitespace-stripped digests, documents) of a shard split at EOT."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_file)
    eot = tok.token_to_id("<|endoftext|>")
    m = np.asarray(np.memmap(shard, dtype=np.uint16, mode="r"))
    ends = np.flatnonzero(m == eot)
    starts = np.concatenate([[0], ends[:-1] + 1])
    exact, stripped = set(), set()
    for i in range(0, len(ends), 4096):
        pieces = [m[s:e].tolist() for s, e in zip(starts[i:i + 4096], ends[i:i + 4096])]
        for t in tok.decode_batch(pieces):
            exact.add(_digest(t))
            stripped.add(_ws_digest(t))
    return exact, stripped, len(ends)


def _init(tok_file, threads, exact, stripped, batch_rows):
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from tokenizers import Tokenizer
    _G["tok"] = Tokenizer.from_file(tok_file)
    _G["eot"] = _G["tok"].token_to_id("<|endoftext|>")
    _G["exact"], _G["stripped"], _G["batch_rows"] = exact, stripped, batch_rows


def tokenize_file(job):
    src, part = job
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    tok, eot = _G["tok"], _G["eot"]
    pf = pq.ParquetFile(src)
    cols = ["text"] + (["token_count"] if "token_count" in pf.schema_arrow.names else [])
    rows = kept = drop_ws = drop_exact = toks = gpt2 = rt_ok = rt_n = 0
    doc_bytes = []
    t0 = time.time()
    with open(part + ".bin.tmp", "wb") as f:
        for batch in pf.iter_batches(batch_size=_G["batch_rows"], columns=cols):
            if "token_count" in cols:
                gpt2 += int(pc.sum(batch.column("token_count")).as_py() or 0)
            keep = []
            for t in batch.column("text").to_pylist():
                rows += 1
                if _ws_digest(t) in _G["stripped"]:
                    drop_ws += 1
                    drop_exact += _digest(t) in _G["exact"]
                    continue
                keep.append(t)
            if not keep:
                continue
            encs = tok.encode_batch(keep)
            lens = [len(e.ids) + 1 for e in encs]
            out = np.empty(sum(lens), dtype=np.uint16)
            pos = 0
            for t, e, n in zip(keep, encs, lens):
                out[pos:pos + n - 1] = e.ids
                out[pos + n - 1] = eot
                pos += n
                doc_bytes.append(len(t.encode("utf-8")))
                if rt_n < 20:
                    rt_n += 1
                    rt_ok += tok.decode(e.ids) == t
            out.tofile(f)
            kept += len(keep)
            toks += len(out)
    np.asarray(doc_bytes, dtype=np.uint32).tofile(part + ".bytes.tmp")
    os.replace(part + ".bin.tmp", part + ".bin")
    os.replace(part + ".bytes.tmp", part + ".bytes")
    st = dict(src=src, part=part, rows=rows, kept=kept, excluded_ws=drop_ws, excluded_exact=drop_exact,
              tokens=toks, utf8_bytes=int(sum(doc_bytes)), parquet_gpt2_tokens=gpt2,
              roundtrip=[rt_ok, rt_n], seconds=round(time.time() - t0, 1))
    with open(part + ".json", "w") as fh:
        json.dump(st, fh)
    return st


def merge(parts, out, eot, shard_tokens):
    """Cut the parts (in order) into K shards at document ends; return per-shard stats."""
    T = sum(p["tokens"] for p in parts)
    K = max(2, round(T / shard_tokens))
    bounds = [(T * (i + 1)) // K for i in range(K)]
    shards, si, gpos = [], 0, 0
    cur = dict(tokens=0, docs=0, utf8_bytes=0)
    path = lambda i: os.path.join(out, f"shard_{i:04d}.bin")
    f = open(path(0) + ".tmp", "wb")

    def write(seg, db, di):
        nd = int(np.count_nonzero(seg == eot))
        seg.tofile(f)
        cur["tokens"] += len(seg)
        cur["docs"] += nd
        cur["utf8_bytes"] += int(db[di:di + nd].sum())
        return di + nd

    for p in parts:
        m = np.memmap(p["part"] + ".bin", dtype=np.uint16, mode="r")
        db = np.fromfile(p["part"] + ".bytes", dtype=np.uint32).astype(np.int64)
        di, pos = 0, 0
        while pos < len(m):
            chunk = np.asarray(m[pos:pos + CHUNK])
            off = 0
            while si < K - 1:
                lo = max(off, bounds[si] - gpos - 1)
                if lo >= len(chunk):
                    break
                hits = np.flatnonzero(chunk[lo:] == eot)
                if len(hits) == 0:
                    break
                cut = lo + int(hits[0]) + 1
                di = write(chunk[off:cut], db, di)
                f.close()
                os.replace(path(si) + ".tmp", path(si))
                shards.append(dict(cur))
                print(f"  shard_{si:04d}: {cur['tokens'] / 1e6:.1f}M tokens, {cur['docs']} docs", flush=True)
                si, off = si + 1, cut
                cur = dict(tokens=0, docs=0, utf8_bytes=0)
                f = open(path(si) + ".tmp", "wb")
            di = write(chunk[off:], db, di)
            gpos += len(chunk)
            pos += len(chunk)
        assert di == len(db), f"{p['part']}: {di} document ends vs {len(db)} documents"
        del m
        for ext in (".bin", ".bytes", ".json"):
            os.remove(p["part"] + ext)
    f.close()
    assert si == K - 1 and cur["tokens"] > 0, f"merge ended at shard {si} of {K}"
    os.replace(path(si) + ".tmp", path(si))
    shards.append(dict(cur))
    print(f"  shard_{si:04d}: {cur['tokens'] / 1e6:.1f}M tokens, {cur['docs']} docs (validation)", flush=True)
    return shards


def build(a):
    t0 = time.time()
    out = a.out
    if os.path.isdir(out) and any(n.startswith("shard_") or n == "meta.json" for n in os.listdir(out)):
        raise SystemExit(f"refusing: {out} already holds shards or meta.json")
    parts_dir = os.path.join(out, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    files = sorted(p for p in glob.glob(os.path.join(a.parquet, "**", "*.parquet"), recursive=True)
                   if "/.cache/" not in p)
    man_path = os.path.join(a.parquet, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else None
    if man:
        want = {os.path.join(a.parquet, x["path"]): x["size"] for x in man["files"]}
        got = {p: os.path.getsize(p) for p in files}
        if want != got:
            raise SystemExit(f"parquet files differ from {man_path}: "
                             f"{len(want)} listed vs {len(got)} on disk")
    if not files:
        raise SystemExit(f"no parquet files under {a.parquet}")
    tok_file = os.path.join(a.tokenizer, "tokenizer.json")
    sha = hashlib.sha256(open(tok_file, "rb").read()).hexdigest()
    tman_path = os.path.join(a.tokenizer, "manifest.json")
    tman = json.load(open(tman_path)) if os.path.exists(tman_path) else {}
    from tokenizers import Tokenizer
    eot = Tokenizer.from_file(tok_file).token_to_id("<|endoftext|>")
    exact, stripped, n_ex = exclude_sets(a.exclude_shard, tok_file) if a.exclude_shard else (set(), set(), 0)
    print(f"{len(files)} parquet files; exclude set: {n_ex} documents "
          f"({len(stripped)} distinct whitespace-stripped) from {a.exclude_shard}", flush=True)
    jobs = [(src, os.path.join(parts_dir, f"part_{i:04d}")) for i, src in enumerate(files)]
    todo = sorted((j for j in jobs if not os.path.exists(j[1] + ".json")), key=lambda j: -os.path.getsize(j[0]))
    print(f"tokenizing {len(todo)} files ({len(jobs) - len(todo)} done earlier): "
          f"{a.workers} workers x {a.threads} threads", flush=True)
    t1, done_tok = time.time(), 0
    if todo:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                          initargs=(tok_file, a.threads, exact, stripped, a.batch_rows)) as pool:
            for k, r in enumerate(pool.imap_unordered(tokenize_file, todo), 1):
                done_tok += r["tokens"]
                print(f"  [{k}/{len(todo)}] {os.path.basename(r['src'])}: {r['rows']} rows, {r['kept']} kept, "
                      f"{r['excluded_ws']} excluded ({r['excluded_exact']} exact), {r['tokens'] / 1e6:.0f}M tokens, "
                      f"rt {r['roundtrip'][0]}/{r['roundtrip'][1]}, {r['seconds']} s | "
                      f"{done_tok / max(1e-9, time.time() - t1) / 1e6:.1f}M tok/s", flush=True)
    stats = [json.load(open(p + ".json")) for _, p in jobs]
    t2 = time.time()
    print(f"merging {sum(s['tokens'] for s in stats) / 1e9:.2f}B tokens", flush=True)
    shards = merge(stats, out, eot, a.shard_tokens)
    os.rmdir(parts_dir)
    meta = {"tokenizer": os.path.basename(a.tokenizer.rstrip("/")), "tokenizer_path": a.tokenizer,
            "tokenizer_sha256": sha, "vocab_size": tman.get("vocab_size", 32768), "eot": eot, "dtype": "uint16",
            "source": ({k: man[k] for k in ("repo", "revision", "subdir")} if man else None),
            "parquet_dir": a.parquet, "parquet_files": len(files),
            "shards": len(shards), "shard_tokens_target": a.shard_tokens,
            "total_tokens": int(sum(s["tokens"] for s in shards)),
            "per_shard_tokens": [s["tokens"] for s in shards],
            "per_shard_docs": [s["docs"] for s in shards],
            "per_shard_utf8_bytes": [s["utf8_bytes"] for s in shards],
            "docs": int(sum(s["kept"] for s in stats)), "rows_read": int(sum(s["rows"] for s in stats)),
            "utf8_bytes": int(sum(s["utf8_bytes"] for s in stats)),
            "exclude_shard": a.exclude_shard, "exclude_shard_docs": n_ex,
            "excluded_docs_whitespace_match": int(sum(s["excluded_ws"] for s in stats)),
            "excluded_docs_exact_match": int(sum(s["excluded_exact"] for s in stats)),
            "parquet_gpt2_token_count": int(sum(s["parquet_gpt2_tokens"] for s in stats)),
            "roundtrip": [int(sum(s["roundtrip"][0] for s in stats)), int(sum(s["roundtrip"][1] for s in stats))],
            "budget_note": "no gpt2_total_tokens on purpose: pass --total-tokens explicitly",
            "seconds": {"tokenize": round(t2 - t1), "merge": round(time.time() - t2), "total": round(time.time() - t0)}}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"DONE: {meta['rows_read']} rows, {meta['docs']} kept, "
          f"{meta['excluded_docs_whitespace_match']} excluded ({meta['excluded_docs_exact_match']} exact); "
          f"{meta['total_tokens'] / 1e9:.2f}B tokens in {meta['shards']} shards "
          f"(validation shard_{meta['shards'] - 1:04d}, {shards[-1]['tokens'] / 1e6:.1f}M); "
          f"parquet GPT-2 count {meta['parquet_gpt2_token_count'] / 1e9:.2f}B; "
          f"round trip {meta['roundtrip'][0]}/{meta['roundtrip'][1]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--out", required=True)
    d.add_argument("--revision", required=True)
    d.add_argument("--repo", default="HuggingFaceFW/fineweb")
    d.add_argument("--subdir", default="sample/100BT")
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--list-only", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--parquet", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--exclude-shard", default=None)
    b.add_argument("--shard-tokens", type=int, default=250_000_000)
    b.add_argument("--workers", type=int, default=24)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--batch-rows", type=int, default=8192)
    a = ap.parse_args()
    download(a) if a.cmd == "download" else build(a)


if __name__ == "__main__":
    main()
