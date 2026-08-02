"""Multi-turn evaluation: does the multi-turn SFT (-it-mt) actually
beat the base -it at cross-turn memory / instruction persistence?

Scripted dialogs, each with a PROBE final turn whose correct answer
requires info stated in an EARLY turn. Score = fraction of probes whose
generated response contains the required back-reference (keyword match,
generous). Greedy decode (deterministic). Run on both ckpts, compare.

Usage: python eval_multiturn_stilt.py <ckpt.pt> [--system-native]
"""
import argparse
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

# each: (list of user turns, index of probe turn, keywords any-of = pass)
CASES = [
    (["My name is Alex.", "Tell me a fun fact.", "What is my name?"],
     2, ["alex"]),
    (["My favorite color is blue.", "Recommend a hobby.",
      "What color should I paint my room?"], 2, ["blue"]),
    (["I'm allergic to peanuts.", "Suggest a snack for me."],
     1, ["allerg", "no peanut", "without peanut", "peanut-free",
         "avoid peanut"]),
    (["I have three apples.", "I eat one apple.",
      "How many apples do I have now?"], 2, ["two", "2"]),
    (["I live in Paris.", "What's a good book?",
      "What city do I live in?"], 2, ["paris"]),
    (["Remember the code word is banana.", "How are you?",
      "What was the code word?"], 2, ["banana"]),
    (["My dog is named Rex.", "Do you like animals?",
      "What is my dog's name?"], 2, ["rex"]),
    (["I am a vegetarian.", "Suggest a dinner."],
     1, ["vegetarian", "veggie", "no meat", "meat-free", "plant"]),
]


def load(ckpt):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    m = GPT(cfg).to(DEVICE)
    m.load_state_dict(blob["model"])
    m.eval()
    return m, cfg


def gen(model, tok, hist, ctx, max_new=60):
    ids = tok(hist, return_tensors="pt").input_ids.to(DEVICE)
    cur = ids
    with torch.no_grad():
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(cur[:, -ctx:])
            nx = lg[0, -1].argmax()
            if nx.item() == tok.eos_token_id:
                break
            cur = torch.cat([cur, nx[None, None]], 1)
    return tok.decode(cur[0, ids.shape[1]:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--system-native", action="store_true")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model, cfg = load(args.ckpt)
    sys_pre = S + SYS + "\n"

    npass = 0
    for turns, probe_i, keys in CASES:
        hist = sys_pre
        resp = ""
        for i, ut in enumerate(turns):
            hist += U + ut + "\n"
            if i == probe_i:
                hist += A
                resp = gen(model, tok, hist, cfg.ctx)
                break
            else:
                hist += A
                r = gen(model, tok, hist, cfg.ctx, max_new=40)
                hist += r + "<|endoftext|>"
        low = resp.lower()
        ok = any(k in low for k in keys)
        npass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] probe={turns[probe_i][:40]!r}"
              f" -> {resp[:70]!r}", flush=True)
    print(f"SUMMARY multiturn back-ref recall: {npass}/{len(CASES)} "
          f"= {100*npass/len(CASES):.0f}%  ({args.ckpt.split('/')[-1]})",
          flush=True)


if __name__ == "__main__":
    main()
