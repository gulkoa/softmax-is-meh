"""prepare_fineweb_tok32k.py build on synthetic parquet with the real
stilt-tok-32k-v1 tokenizer: documents of the old validation shard are dropped
on exact and whitespace-only matches and nothing else is; one EOT per kept
document; the shards decode to the kept documents in file order; shards are
cut at document ends and balanced; meta.json has the trainer's keys, correct
counts and no gpt2_total_tokens; parts are cleaned up; a second build into the
same directory is refused."""
import json, os, subprocess, sys, tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "triton", "dev_scripts", "prepare_fineweb_tok32k.py")
TOK = "/fs/scratch/PAS2836/alexg/tokenizers/stilt-tok-32k-v1"
WORDS = ["alpha", "beta", "γάμμα", "δέλτα", "naïve", "café", "日本語", "😀", "x=1;", "\tTab",
         "line\nbreak", "  spaced  ", "won't", "(paren)", "1,234.5"]


def main():
    if not os.path.exists(os.path.join(TOK, "tokenizer.json")):
        print("SKIP — tokenizer not on this filesystem")
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(TOK, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    rng = np.random.default_rng(0)

    def doc(i):
        return f"Doc {i}: " + " ".join(rng.choice(WORDS, size=int(rng.integers(5, 60)))) + " ."

    files = [[doc(f * 1000 + i) for i in range(n)] for f, n in enumerate([40, 25, 60])]
    exact = [files[0][3], files[1][10], files[2][59]]
    ws_only = [files[0][7].replace(" .", "."), files[2][0].replace(" ", "  ", 1)]
    assert all(t not in sum(files, []) for t in ws_only)
    absent = ["not in the data at all", "neither is this one"]
    dropped = set(exact) | {files[0][7], files[2][0]}
    kept = [t for f in files for t in f if t not in dropped]

    with tempfile.TemporaryDirectory() as d:
        pqdir = os.path.join(d, "pq", "sample", "100BT")
        os.makedirs(pqdir)
        for k, texts in enumerate(files):
            pq.write_table(pa.table({"text": texts, "token_count": [len(t) for t in texts]}),
                           os.path.join(pqdir, f"{k:03d}_00000.parquet"), row_group_size=7)
        ids = []
        for t in exact + ws_only + absent:
            ids += tok.encode(t).ids + [eot]
        old = os.path.join(d, "old_val.bin")
        np.asarray(ids, dtype=np.uint16).tofile(old)
        out = os.path.join(d, "out")
        cmd = [sys.executable, SCRIPT, "build", "--parquet", os.path.join(d, "pq"), "--out", out,
               "--tokenizer", TOK, "--exclude-shard", old, "--shard-tokens", "1500",
               "--workers", "2", "--threads", "1", "--batch-rows", "5"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

        meta = json.load(open(os.path.join(out, "meta.json")))
        shards = sorted(n for n in os.listdir(out) if n.startswith("shard_"))
        assert not os.path.exists(os.path.join(out, "parts")), "parts left behind"
        arrs = [np.fromfile(os.path.join(out, s), dtype=np.uint16) for s in shards]
        assert all(len(x) and x[-1] == eot for x in arrs), "a shard does not end with EOT"
        allids = np.concatenate(arrs)
        ends = np.flatnonzero(allids == eot)
        starts = np.concatenate([[0], ends[:-1] + 1])
        texts = [tok.decode(allids[s:e].tolist()) for s, e in zip(starts, ends)]
        assert texts == kept, "decoded documents differ from the kept documents in file order"

        T, K = len(allids), max(2, round(len(allids) / 1500))
        assert meta["shards"] == len(shards) == K, (meta["shards"], len(shards), K)
        longest = max(e - s + 1 for s, e in zip(starts, ends))
        assert all(abs(len(x) - T / K) <= longest + 1 for x in arrs), [len(x) for x in arrs]
        assert meta["total_tokens"] == T and meta["per_shard_tokens"] == [len(x) for x in arrs]
        assert meta["docs"] == len(kept) == sum(meta["per_shard_docs"])
        assert meta["per_shard_docs"] == [int(np.count_nonzero(x == eot)) for x in arrs]
        assert meta["utf8_bytes"] == sum(len(t.encode()) for t in kept) == sum(meta["per_shard_utf8_bytes"])
        assert meta["rows_read"] == sum(len(f) for f in files)
        assert meta["excluded_docs_whitespace_match"] == 5 and meta["excluded_docs_exact_match"] == 3, meta
        assert meta["exclude_shard_docs"] == 7
        assert meta["parquet_gpt2_token_count"] == sum(len(t) for f in files for t in f)
        assert meta["roundtrip"][0] == meta["roundtrip"][1] > 0
        assert meta["eot"] == eot == 0 and meta["vocab_size"] == 32768
        assert meta["tokenizer"] == "stilt-tok-32k-v1" and meta["tokenizer_path"] == TOK
        assert len(meta["tokenizer_sha256"]) == 64 and "gpt2_total_tokens" not in meta

        r2 = subprocess.run(cmd, capture_output=True, text=True)
        assert r2.returncode != 0 and "refusing" in r2.stdout + r2.stderr, "overwrite not refused"
    print(f"OK — {len(kept)} documents kept, 5 excluded (3 exact), {K} shards of ~{T // K} tokens")


if __name__ == "__main__":
    main()
