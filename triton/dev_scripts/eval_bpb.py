"""Bits per byte on the held-out shards — the tokenizer-independent LM metric
for the tokenizer pilot (plan 2026-10-14-plan-tokenizer-32k).

Takes the first --docs documents of each source's LAST shard (the trainer's
validation shard — the same documents under every tokenizer, because the
re-tokenizer keeps shard indices and document order), joins them with the
tokenizer's EOT exactly as the shard does, scores non-overlapping windows of
ctx+1 tokens with no context carried across windows, and reports per source
the token perplexity, the bytes per token and the bits per byte
(sum NLL / (ln 2 * utf-8 bytes of the scored target tokens)), plus the
70/15/15-weighted BPB. GPU (falls back to CPU).

  python eval_bpb.py <ckpt.pt> [--docs 1500] [--mix web=dir:w,math=dir:w,code=dir:w] [--bs 16]
(default --mix = the checkpoint's own data_mix, i.e. the shard dirs it trained on)
"""
import argparse, math, os, sys
from types import SimpleNamespace
import numpy as np
import torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CHUNK = 20_000_000


def first_docs(path, eot, n_docs):
    """ids of the first n_docs documents of a shard, each followed by EOT (as stored)."""
    m = np.memmap(path, dtype=np.uint16, mode="r")
    out, got, pos = [], 0, 0
    while got < n_docs and pos < len(m):
        chunk = np.asarray(m[pos:pos + CHUNK]).astype(np.int64)
        cuts = np.flatnonzero(chunk == eot)
        if len(cuts) == 0:
            raise RuntimeError("no document end found in the first chunk; raise CHUNK")
        take = min(n_docs - got, len(cuts))
        end = cuts[take - 1] + 1                       # through the take-th EOT
        out.append(chunk[:end]); got += take; pos += end
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--docs", type=int, default=1500)
    ap.add_argument("--mix", default=None)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(cfg).to(dev).eval()
    model.load_state_dict(blob["model"])
    tname = getattr(cfg, "tokenizer", "gpt2") or "gpt2"
    tok = AutoTokenizer.from_pretrained(getattr(cfg, "tokenizer_path", None) or "gpt2" if tname != "gpt2" else "gpt2")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    ctx = cfg.ctx
    spec = a.mix or cfg.data_mix
    weights = {"web": 0.7, "math": 0.15, "code": 0.15}
    print(f"ckpt {os.path.basename(a.ckpt)} | tokenizer {tname} (EOT {eot}) | ctx {ctx} | {a.docs} docs per source", flush=True)
    print(f"{'source':<6} {'tokens':>9} {'bytes':>10} {'B/tok':>6} {'token ppl':>10} {'BPB':>7}")
    rows = {}
    for part in spec.split(","):
        name, rest = part.split("=", 1) if "=" in part else ("web", part)
        d = rest.rsplit(":", 1)[0] if ":" in rest else rest
        last = sorted(f for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".bin"))[-1]
        ids = first_docs(f"{d}/{last}", eot, a.docs)
        n_win = (len(ids) - 1) // ctx
        xs = torch.from_numpy(ids[: n_win * ctx + 1])
        nll, ntok, nbytes = 0.0, 0, 0
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            for b in range(0, n_win, a.bs):
                idx = torch.arange(b, min(b + a.bs, n_win))
                x = torch.stack([xs[i * ctx: i * ctx + ctx] for i in idx]).to(dev)
                y = torch.stack([xs[i * ctx + 1: i * ctx + ctx + 1] for i in idx]).to(dev)
                _, loss = model(x, y)
                nll += float(loss) * y.numel(); ntok += y.numel()
                for row in y.tolist():
                    nbytes += len(tok.decode(row, skip_special_tokens=True).encode("utf-8"))
        bpb = nll / (math.log(2) * nbytes)
        rows[name] = dict(tokens=ntok, bytes=nbytes, bpt=nbytes / ntok, ppl=math.exp(nll / ntok), bpb=bpb)
        print(f"{name:<6} {ntok:>9} {nbytes:>10} {nbytes/ntok:>6.3f} {math.exp(nll/ntok):>10.3f} {bpb:>7.4f}", flush=True)
    mix = sum(weights.get(k, 0) * r["bpb"] for k, r in rows.items()) / sum(weights.get(k, 0) for k in rows)
    print(f"mix-weighted BPB (70/15/15): {mix:.4f}", flush=True)


if __name__ == "__main__":
    main()
