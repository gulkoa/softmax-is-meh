"""M3 eval for the GPT-2-style runs: held-out FineWeb ppl, WikiText-103
ppl, LAMBADA last-word accuracy, and sample completions.

Usage: python eval_gpt2_stieltjes.py <ckpt.pt> [<ckpt2.pt> ...]
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT, Shards, FW_DIR  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEVICE = torch.device("cuda")
PROMPTS = [
    "The most surprising thing about deep learning is",
    "In a small village by the sea, there lived",
    "The recipe requires three eggs, a cup of flour, and",
]


@torch.no_grad()
def fineweb_ppl(model, ctx, iters=100, bs=16):
    val = Shards("val")
    rng = np.random.default_rng(123)
    tot = 0.0
    for _ in range(iters):
        x, y = val.batch(bs, ctx, rng, DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item()
    return math.exp(tot / iters)


@torch.no_grad()
def wikitext_ppl(model, tok, ctx):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())[:2_000_000]
    ids = tok(text, add_special_tokens=False, return_tensors="pt"
              ).input_ids[0].to(DEVICE)
    nll, count = 0.0, 0
    for lo in range(0, len(ids) - ctx - 1, ctx):
        x = ids[lo:lo + ctx][None]
        y = ids[lo + 1:lo + ctx + 1][None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        nll += loss.item() * ctx
        count += ctx
    return math.exp(nll / count)


@torch.no_grad()
def lambada_acc(model, tok, n=1000):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    correct = 0
    n = min(n, len(ds))
    for i in range(n):
        text = ds[i]["text"]
        prefix, target = text.rsplit(" ", 1)
        ids = tok(prefix, add_special_tokens=False,
                  return_tensors="pt").input_ids.to(DEVICE)
        tgt_ids = tok(" " + target, add_special_tokens=False).input_ids
        gen = []
        cur = ids
        for _ in range(len(tgt_ids)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(cur[:, -1024:])
            nxt = logits[0, -1].argmax().item()
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], 1)
        correct += gen == tgt_ids
    return correct / n


@torch.no_grad()
def completions(model, tok):
    outs = []
    for p in PROMPTS:
        cur = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        for _ in range(40):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(cur[:, -1024:])
            probs = F_softmax_top(logits[0, -1], k=40)
            nxt = torch.multinomial(probs, 1)
            cur = torch.cat([cur, nxt[None]], 1)
        outs.append(tok.decode(cur[0]))
    return outs


def F_softmax_top(logits, k):
    v, ix = torch.topk(logits.float(), k)
    p = torch.softmax(v, -1)
    full = torch.zeros_like(logits, dtype=torch.float32)
    full[ix] = p
    return full


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(0)
    for path in sys.argv[1:]:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SimpleNamespace(**blob["args"])
        model = GPT(cfg).to(DEVICE)
        model.load_state_dict(blob["model"])
        model.eval()
        print(f"\n===== {os.path.basename(path)} (step {blob['step']}) =====",
              flush=True)
        print(f"fineweb-val ppl : {fineweb_ppl(model, cfg.ctx):8.2f}", flush=True)
        print(f"wikitext103 ppl : {wikitext_ppl(model, tok, cfg.ctx):8.2f}",
              flush=True)
        print(f"lambada acc     : {lambada_acc(model, tok):8.3f}", flush=True)
        for c in completions(model, tok):
            print(f"  >> {c!r}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
