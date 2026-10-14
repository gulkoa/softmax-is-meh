"""Item 6 pilot (login node, CPU): how much would a 32k byte-level BPE trained
on OUR mix buy over GPT-2's 50,257-token BPE?

Text comes from the existing uint16 shards, decoded with the GPT-2 tokenizer
(byte-level BPE decode is lossless): training text from shard_0000 of each
source, held-out text from each source's LAST shard (the trainer's val
shard). Candidates are trained with HF `tokenizers` on a 70/15/15
web/math/code sample and scored on the held-out text by bytes per token
(higher = fewer tokens per document = more text per fixed token budget).

Output: results/tokenizer_pilot_<date>.json + a markdown table on stdout.
"""
import argparse, json, os, re, time, datetime as dt
import numpy as np
os.environ.setdefault("HF_HOME", "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/hf_cache")
os.environ.setdefault("RAYON_NUM_THREADS", "32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex

B = "/fs/scratch/PAS2836/alexg"
SOURCES = {"web": f"{B}/fineweb_edu_10bt", "math": f"{B}/finemath_4plus", "code": f"{B}/codeparrot_py"}
MIX = {"web": 0.70, "math": 0.15, "code": 0.15}
EOT = 50256
GPT2_PAT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
CL100K_PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+")
DIGIT1_PAT = CL100K_PAT.replace(r"\p{N}{1,3}", r"\p{N}")
SPECIAL = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|eot|>", "<|pad|>"]


def docs_from_shard(path, tok, n_tokens, from_end=False):
    """Decode ~n_tokens of whole documents from a shard into a list of strings."""
    m = np.memmap(path, dtype=np.uint16, mode="r")
    ids = np.asarray(m[-n_tokens:] if from_end else m[:n_tokens]).astype(np.int64)
    cuts = np.flatnonzero(ids == EOT)
    docs, start = [], 0
    for c in cuts:
        if c > start:
            docs.append(ids[start:c].tolist())
        start = c + 1
    return tok.batch_decode(docs, skip_special_tokens=True)


def train_bpe(texts, vocab_size, pattern):
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t0 = time.time()
    tk.train_from_iterator(texts, trainer=tr, length=len(texts))
    return tk, time.time() - t0


def score(encode, texts):
    nb = sum(len(t.encode("utf-8")) for t in texts)
    t0 = time.time()
    nt = sum(len(ids) for ids in encode(texts))
    return dict(bytes=nb, tokens=nt, bytes_per_token=nb / nt, mb_per_s=nb / 1e6 / (time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=float, default=60e6, help="web tokens for BPE training; math/code scaled by MIX")
    ap.add_argument("--eval-tokens", type=float, default=4e6, help="held-out tokens per source")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    train_texts, eval_texts = [], {}
    for src, d in SOURCES.items():
        shards = sorted(f for f in os.listdir(d) if f.startswith("shard_"))
        n_train = int(a.train_tokens * MIX[src] / MIX["web"])
        t0 = time.time()
        train_texts += docs_from_shard(f"{d}/{shards[0]}", gpt2, n_train)
        eval_texts[src] = docs_from_shard(f"{d}/{shards[-1]}", gpt2, int(a.eval_tokens), from_end=True)
        print(f"{src}: {n_train/1e6:.0f}M train tokens from {shards[0]}, {len(eval_texts[src])} held-out docs "
              f"from {shards[-1]} ({time.time()-t0:.0f}s)", flush=True)
    print(f"training sample: {len(train_texts)} docs, {sum(len(t) for t in train_texts)/1e6:.0f}M chars", flush=True)
    res = {"gpt2-50257": {"train_s": 0.0}}
    enc_gpt2 = lambda texts: gpt2(texts, add_special_tokens=False)["input_ids"]
    for src, texts in eval_texts.items():
        res["gpt2-50257"][src] = score(enc_gpt2, texts)
    cands = [("bpe32k-gpt2re", 32768, GPT2_PAT), ("bpe32k-cl100kre", 32768, CL100K_PAT),
             ("bpe32k-digit1", 32768, DIGIT1_PAT), ("bpe16k-cl100kre", 16384, CL100K_PAT),
             ("bpe50k-cl100kre", 50257, CL100K_PAT), ("bpe64k-cl100kre", 65536, CL100K_PAT)]
    for name, vs, pat in cands:
        tk, secs = train_bpe(train_texts, vs, pat)
        enc = lambda texts, tk=tk: [e.ids for e in tk.encode_batch(texts)]
        res[name] = {"train_s": secs, "vocab": tk.get_vocab_size()}
        for src, texts in eval_texts.items():
            res[name][src] = score(enc, texts)
        # round-trip check on a few held-out docs
        rt = all(tk.decode(tk.encode(t).ids) == t for t in eval_texts["code"][:20] + eval_texts["web"][:20])
        res[name]["roundtrip_ok"] = rt
        print(f"{name}: trained in {secs:.0f}s, vocab {tk.get_vocab_size()}, roundtrip {rt}", flush=True)
        if name == "bpe32k-cl100kre":
            tk.save(f"/users/PAS2402/alexg/softmax/softmax-is-meh/results/tokenizer_pilot_bpe32k_cl100kre.json")
    # digit handling probe
    probes = ["In 2024 the value was 1234567.89 and 3.14159.", "x = [10, 200, 3000]; y = 0.001",
              "    for i in range(10):\n        total += values[i]\n"]
    res["probes"] = {}
    for name in ("gpt2-50257", "bpe32k-cl100kre", "bpe32k-digit1"):
        if name == "gpt2-50257":
            res["probes"][name] = [gpt2.tokenize(p) for p in probes]
        else:
            pass
    out = a.out or (f"/users/PAS2402/alexg/softmax/softmax-is-meh/results/tokenizer_pilot_{dt.date.today().isoformat()}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("\n| tokenizer | vocab | train s | web B/tok | math B/tok | code B/tok | mix B/tok | tokens vs GPT-2 (mix) |")
    print("|---|---|---|---|---|---|---|---|")
    def mix_bpt(r):
        bt = sum(MIX[s] * r[s]["bytes"] / r[s]["bytes_per_token"] for s in SOURCES)   # tokens weighted by mix
        bb = sum(MIX[s] * r[s]["bytes"] for s in SOURCES)
        return bb / bt
    base = mix_bpt(res["gpt2-50257"])
    for name, r in res.items():
        if name == "probes": continue
        print(f"| {name} | {r.get('vocab', 50257)} | {r['train_s']:.0f} | {r['web']['bytes_per_token']:.3f} | "
              f"{r['math']['bytes_per_token']:.3f} | {r['code']['bytes_per_token']:.3f} | {mix_bpt(r):.3f} | "
              f"{base/mix_bpt(r):.3f}× |")
    print("wrote", out)


if __name__ == "__main__":
    main()
