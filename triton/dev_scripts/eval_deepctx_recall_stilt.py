"""Deep-context recall: can the model retrieve a fact stated D tokens ago?

The nope-it artifact's raison d'être: it trained on UNTRIMMED
conversations at ctx 2048 (positional models cap at 1024 and are served
with a sliding window, so facts beyond the window are unrecoverable by
construction). This measures recall of an early-turn fact at controlled
token depths, sweeping PAST the positional window.

Per (fact, depth): system + fact turn + filler Q/A turns until the
probe sits ~depth tokens after the fact + probe turn; greedy 24 tokens;
pass = any keyword in the response. Positional ckpts generate with
their trained sliding window (honest serving); position-free ckpts
(nope/rope) use the full history.

Usage: python eval_deepctx_recall_stilt.py <ckpt.pt> [--out results.json]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT, FW_DIR  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
from transformers import AutoTokenizer  # noqa: E402

DEVICE = torch.device("cuda")
S, U, A = "<|system|>\n", "<|user|>\n", "<|assistant|>\n"
SYS = "You are stilt, a helpful AI assistant"

FACTS = [
    ("My name is Alex.", "What is my name?", ["alex"]),
    ("My favorite color is blue.", "What is my favorite color?", ["blue"]),
    ("Remember the code word is banana.", "What was the code word?",
     ["banana"]),
    ("My dog is named Rex.", "What is my dog's name?", ["rex"]),
    ("I live in Paris.", "What city do I live in?", ["paris"]),
    ("My lucky number is 17.", "What is my lucky number?",
     ["17", "seventeen"]),
]

FILLERS = [
    ("Tell me a fun fact about space.",
     "The Sun contains about 99.8% of the mass in our solar system."),
    ("What is a good breakfast?",
     "Oatmeal with fruit is a simple, filling breakfast."),
    ("How do plants grow?",
     "Plants use sunlight, water, and nutrients from soil to grow."),
    ("Name a famous painter.",
     "Vincent van Gogh is one of the most famous painters."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("How does rain form?",
     "Water evaporates, condenses into clouds, and falls as rain."),
    ("Suggest a light exercise.",
     "A brisk 20-minute walk is a great light exercise."),
    ("What do bees make?", "Bees make honey from flower nectar."),
    ("Name a musical instrument.",
     "The piano is a widely played musical instrument."),
    ("What season comes after summer?", "Autumn comes after summer."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What is bread made from?",
     "Bread is mostly made from flour, water, and yeast."),
]

DEPTHS = [600, 1200, 1800, 2400, 3000]


def build_prompt(tok, fact, probe, depth):
    head = S + SYS + "\n" + U + fact + "\n" + A + "Got it!<|endoftext|>"
    i = 0
    body = ""
    while len(tok(head + body, add_special_tokens=False).input_ids) < depth:
        q, a = FILLERS[i % len(FILLERS)]
        body += U + q + "\n" + A + a + "<|endoftext|>"
        i += 1
    return head + body + U + probe + "\n" + A


@torch.no_grad()
def generate(model, tok, prompt, window):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    cur = ids
    for _ in range(24):
        x = cur if window is None else cur[:, -window:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(x)
        nx = lg[0, -1].argmax()
        if nx.item() == tok.eos_token_id:
            break
        cur = torch.cat([cur, nx[None, None]], 1)
    return tok.decode(cur[0, ids.shape[1]:]).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    model.eval()
    tok = AutoTokenizer.from_pretrained("gpt2")

    posfree = getattr(cfg, "nope", False) or getattr(cfg, "rope", False)
    window = None if posfree else cfg.ctx
    print(f"{os.path.basename(args.ckpt)}: position-free={posfree} "
          f"window={window}", flush=True)

    rows = []
    for depth in DEPTHS:
        hits = 0
        for fact, probe, keys in FACTS:
            prompt = build_prompt(tok, fact, probe, depth)
            n_tok = len(tok(prompt, add_special_tokens=False).input_ids)
            resp = generate(model, tok, prompt, window)
            hit = any(k in resp for k in keys)
            hits += hit
            print(f"  d={depth} ({n_tok} tok) {probe[:24]!r}: "
                  f"{'HIT ' if hit else 'MISS'} {resp[:60]!r}", flush=True)
        rows.append({"depth": depth, "recall": hits / len(FACTS)})
        print(f"DEPTH {depth}: recall {hits}/{len(FACTS)}", flush=True)

    out = args.out or args.ckpt.replace(".pt", "_deepctx_recall.json")
    json.dump({"ckpt": args.ckpt, "posfree": posfree, "window": window,
               "rows": rows}, open(out, "w"), indent=2)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
