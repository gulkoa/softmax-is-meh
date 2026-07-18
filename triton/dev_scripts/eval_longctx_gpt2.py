"""Long-context NL comparison for the trained GPT-2 pair (ctx 1024).

A. Context-utilization: per-token NLL bucketed by available left-context
   (within the 1024 window) on real long documents — does the mechanism
   keep extracting signal from deeper context?
B. Beyond-ctx degradation: ppl at 1024/2048/4096 with saturated learned
   positions (positions clamp at ctx-1) — graceful-degradation slope.
Sources: PG-19 test books + FineWeb-Edu held-out shard.

Usage: python eval_longctx_gpt2.py <ckpt.pt> [<ckpt2.pt> ...]
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT, Shards, FW_DIR  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEVICE = torch.device("cuda")
BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1023)]


def long_token_stream(tok, n_tokens=1_500_000):
    """~1.5M tokens of genuinely long documents (PG-19 test books)."""
    ds = load_dataset("deepmind/pg19", split="test", streaming=True)
    ids = []
    for ex in ds:
        ids.extend(tok(ex["text"], add_special_tokens=False).input_ids)
        if len(ids) >= n_tokens:
            break
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


@torch.no_grad()
def context_utilization(model, ids, ctx=1024, bs=16, max_windows=400):
    """Mean NLL per available-context bucket over non-overlapping windows."""
    sums = np.zeros(len(BUCKETS))
    counts = np.zeros(len(BUCKETS))
    starts = list(range(0, len(ids) - ctx - 1, ctx))[:max_windows]
    for lo in range(0, len(starts), bs):
        batch = starts[lo:lo + bs]
        x = torch.stack([ids[s:s + ctx] for s in batch]).to(DEVICE)
        y = torch.stack([ids[s + 1:s + ctx + 1] for s in batch]).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        nll = nll.float().mean(0).cpu().numpy()      # (ctx,) per-position
        for bi, (a, b) in enumerate(BUCKETS):
            sums[bi] += nll[a:b].sum()
            counts[bi] += (b - a)
    return sums / counts


@torch.no_grad()
def beyond_ctx_ppl(model, ids, lengths=(1024, 2048, 4096), max_tokens=400_000):
    out = {}
    for L in lengths:
        nll, count = 0.0, 0
        for lo in range(0, min(len(ids), max_tokens) - L - 1, L):
            x = ids[lo:lo + L][None].to(DEVICE)
            y = ids[lo + 1:lo + L + 1][None].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            nll += F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   y.view(-1)).item() * L
            count += L
        out[L] = math.exp(nll / count)
    return out


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")
    print("building PG-19 token stream...", flush=True)
    pg = long_token_stream(tok)
    fw = Shards("val")
    fw_ids = torch.from_numpy(
        np.array(fw.sources[0][0][:1_500_000], dtype=np.int64))

    for path in sys.argv[1:]:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SimpleNamespace(**blob["args"])
        model = GPT(cfg).to(DEVICE)
        model.load_state_dict(blob["model"])
        model.eval()
        name = os.path.basename(path)
        print(f"\n===== {name} =====", flush=True)
        for src_name, ids in [("pg19", pg), ("fineweb-val", fw_ids)]:
            util = context_utilization(model, ids, ctx=cfg.ctx)
            buck = "  ".join(f"[{a}-{b}):{u:.4f}"
                             for (a, b), u in zip(BUCKETS, util))
            print(f"  {src_name} NLL by available context: {buck}",
                  flush=True)
            gain = util[0] - util[-1]
            print(f"  {src_name} context gain (bucket0 - bucket4): "
                  f"{gain:.4f} nats", flush=True)
            bp = beyond_ctx_ppl(model, ids)
            print(f"  {src_name} ppl @1024/2048/4096 (pos-saturated): "
                  + " / ".join(f"{bp[L]:.2f}" for L in (1024, 2048, 4096)),
                  flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
