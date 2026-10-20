"""Check a prepare_fineweb_tok32k.py build after it finishes (light enough for a
login node: reads meta.json, the last token of every shard, the first and the
last shard in full, and a few hundred parquet rows).

Always checked:
- meta.json: kept + excluded = rows read; shard tokens and documents sum to
  the totals; shards = max(2, round(total / target)); the round trip passed on
  every sampled document; the trainer's keys are present and
  gpt2_total_tokens is absent (the budget is passed explicitly);
- files: one shard file per meta entry, 2 bytes per token, each ending with
  EOT; no parts directory or .tmp file left;
- the first and the last (validation) shard: the EOT count equals the
  document count and no other special id appears;
- order: the first documents of shard_0000 appear in order among the first
  rows of the first parquet file, and the last document of the validation
  shard is among the last rows of the last file.
The --expect-* flags also pin counts known in advance (a FAIL line each).
Prints one ok/FAIL line per check, then ALL CHECKS PASSED or the number
failed; exit status 1 on any failure.

Usage: check_fineweb_tok32k.py --out DIR --parquet DIR [--expect-rows N] ...
"""
import argparse, glob, json, os, sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the build's output directory (meta.json, shard_*.bin)")
    ap.add_argument("--parquet", required=True, help="the parquet directory the build read")
    ap.add_argument("--expect-revision")
    ap.add_argument("--expect-files", type=int)
    ap.add_argument("--expect-rows", type=int)
    ap.add_argument("--expect-excluded", type=int, help="whitespace-stripped matches (all excluded documents)")
    ap.add_argument("--expect-excluded-exact", type=int)
    ap.add_argument("--expect-exclude-docs", type=int, help="documents in the old validation shard")
    ap.add_argument("--expect-tokenizer-sha", help="prefix of the tokenizer's sha256")
    ap.add_argument("--head-docs", type=int, default=5)
    ap.add_argument("--rows-window", type=int, default=200)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    bad = []

    def check(ok, msg):
        print(("ok   " if ok else "FAIL ") + msg, flush=True)
        if not ok:
            bad.append(msg)

    m = json.load(open(os.path.join(a.out, "meta.json")))
    K, eot = m["shards"], m["eot"]
    ex, exx = m["excluded_docs_whitespace_match"], m["excluded_docs_exact_match"]
    if a.expect_revision:
        check((m.get("source") or {}).get("revision") == a.expect_revision, f"source {m.get('source')}")
    if a.expect_files is not None:
        check(m["parquet_files"] == a.expect_files, f"parquet files {m['parquet_files']}")
    if a.expect_rows is not None:
        check(m["rows_read"] == a.expect_rows, f"rows read {m['rows_read']:,} (expected {a.expect_rows:,})")
    if a.expect_excluded is not None:
        check(ex == a.expect_excluded, f"excluded {ex:,} (expected {a.expect_excluded:,})")
    if a.expect_excluded_exact is not None:
        check(exx == a.expect_excluded_exact, f"excluded exact {exx:,} (expected {a.expect_excluded_exact:,})")
    if a.expect_exclude_docs is not None:
        check(m["exclude_shard_docs"] == a.expect_exclude_docs, f"old validation shard docs {m['exclude_shard_docs']:,}")
    if a.expect_tokenizer_sha:
        check(m["tokenizer_sha256"].startswith(a.expect_tokenizer_sha), f"tokenizer sha {m['tokenizer_sha256'][:12]}")
    check(m["docs"] + ex == m["rows_read"], f"kept {m['docs']:,} + excluded {ex:,} = rows {m['rows_read']:,}")
    check(exx <= ex, "exact matches are a subset of the whitespace matches")
    check(sum(m["per_shard_tokens"]) == m["total_tokens"], f"total tokens {m['total_tokens']:,} = sum of shards")
    check(sum(m["per_shard_docs"]) == m["docs"], "shard documents sum to the kept documents")
    check(len(m["per_shard_tokens"]) == len(m["per_shard_docs"]) == K
          and K == max(2, round(m["total_tokens"] / m["shard_tokens_target"])), f"{K} shards")
    pt = np.asarray(m["per_shard_tokens"])
    print(f"     shard tokens min {pt.min():,} max {pt.max():,}; validation shard_{K - 1:04d}: "
          f"{pt[-1]:,} tokens, {m['per_shard_docs'][-1]:,} documents")
    check(m["roundtrip"][0] == m["roundtrip"][1] > 0, f"round trip {m['roundtrip']}")
    check("gpt2_total_tokens" not in m, "no gpt2_total_tokens key")
    for k in ("tokenizer", "tokenizer_sha256", "vocab_size", "tokenizer_path"):
        check(k in m, f"trainer key {k}")
    check(m["dtype"] == "uint16" and m["vocab_size"] <= 65536, f"dtype {m['dtype']}, vocab {m['vocab_size']}")
    if m.get("parquet_gpt2_token_count"):
        text_tokens = m["total_tokens"] - m["docs"]
        print(f"     tokens per GPT-2 token, no EOT on either side: {text_tokens / m['parquet_gpt2_token_count']:.4f} "
              f"(the GPT-2 count covers every row read, excluded ones too); "
              f"UTF-8 bytes per token, no EOT: {m['utf8_bytes'] / text_tokens:.3f}")

    files = sorted(glob.glob(os.path.join(a.out, "shard_*.bin")))
    check(len(files) == K, f"{len(files)} shard files")
    check(not os.path.exists(os.path.join(a.out, "parts")), "parts directory removed")
    check(not glob.glob(os.path.join(a.out, "*.tmp")), "no .tmp files")
    if len(files) != K:
        print(f"{len(bad)} FAILED")
        sys.exit(1)
    check(all(os.path.getsize(f) == 2 * n for f, n in zip(files, m["per_shard_tokens"])),
          "every shard file is 2 bytes x its token count")
    check(all(np.memmap(f, dtype=np.uint16, mode="r")[-1] == eot for f in files), "every shard ends with EOT")

    tok = Tokenizer.from_file(os.path.join(m["tokenizer_path"], "tokenizer.json"))
    special = np.zeros(max(m["vocab_size"], tok.get_vocab_size()), dtype=bool)
    special[list(tok.get_added_tokens_decoder())] = True
    special[eot] = False
    arr = {}
    for i in sorted({0, K - 1}):
        arr[i] = np.fromfile(files[i], dtype=np.uint16)
        n_eot = int(np.count_nonzero(arr[i] == eot))
        n_other = int(np.count_nonzero(special[arr[i]]))
        check(n_eot == m["per_shard_docs"][i] and n_other == 0,
              f"shard_{i:04d}: {n_eot:,} EOT for {m['per_shard_docs'][i]:,} documents, {n_other} other special ids")

    def docs_of(x, which):
        ends = np.flatnonzero(x == eot)
        starts = np.concatenate([[0], ends[:-1] + 1])
        return [tok.decode(x[starts[j]:ends[j]].tolist(), skip_special_tokens=False) for j in which(len(ends))]

    pqs = sorted(p for p in glob.glob(os.path.join(a.parquet, "**", "*.parquet"), recursive=True)
                 if "/.cache/" not in p)
    check(len(pqs) == m["parquet_files"], f"{len(pqs)} parquet files on disk")
    head = docs_of(arr[0], lambda n: range(min(a.head_docs, n)))
    first = next(pq.ParquetFile(pqs[0]).iter_batches(batch_size=a.rows_window, columns=["text"]))
    rows = iter(first.column("text").to_pylist())
    check(all(any(d == r for r in rows) for d in head),
          f"the first {len(head)} documents of shard_0000 appear in order in {os.path.basename(pqs[0])}")
    last_doc = docs_of(arr[K - 1], lambda n: [n - 1])[0]
    pf, tail, g = pq.ParquetFile(pqs[-1]), [], None
    for g in range(pf.num_row_groups - 1, -1, -1):
        tail = pf.read_row_group(g, columns=["text"]).column("text").to_pylist() + tail
        if len(tail) >= a.rows_window:
            break
    check(last_doc in tail[-a.rows_window:],
          f"the last document of shard_{K - 1:04d} is among the last rows of {os.path.basename(pqs[-1])}")
    print("ALL CHECKS PASSED" if not bad else f"{len(bad)} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
