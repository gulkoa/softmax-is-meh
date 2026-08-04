"""Generation-entropy trajectory across a checkpoint ladder (stj vs twin).

Backs the repositioned mechanism claim (findings/2026-08-03-related-work-
positioning.md § Strengthening experiment): the weighting function's
inductive bias on generation entropy is expressed THROUGH training, not
present at init. This measures mean output-distribution entropy at each
checkpoint of a training-stage ladder (init -> pretrain-final -> -it ->
-it-code-r -> _grpo{100..400}), per arm.

Metric matches the RL-era logs exactly (grpo_code_stilt.py): sample at
temperature 0.8 / top-50 (left-pad EOS), then teacher-forced full-vocab
temperature-1 entropy -(p.log p).sum(-1) averaged over generated tokens
up through the first EOS. The 0.85-vs-3.68 RL-base anchors used MBPP
prompt formatting; pass --prompts-json with those prompts for absolute
comparability, otherwise the builtin fixed sets give a self-consistent
trajectory.

Usage (1h debug GPU, post-contention — do not compete with #35):
  python eval_gen_entropy_stilt.py \
      --ckpt pretrain=/fs/scratch/.../ckpt_gpt2-stj-q4-medium-mix_s0.pt \
      --ckpt it=/fs/scratch/.../ckpt_gpt2-stj-q4-medium-mix_s0-it.pt \
      --ckpt it-code-r=/fs/scratch/.../..._s0-it-code-r.pt \
      --ckpt rl300=/fs/scratch/.../..._grpo300.pt \
      --init-from /fs/scratch/.../ckpt_gpt2-stj-q4-medium-mix_s0.pt \
      --arm stj --out results/gen_entropy_stj.json [--wandb]
Run once per arm; the figure overlays both arms' curves.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from types import SimpleNamespace  # noqa: E402

from train_gpt2_stieltjes import GPT  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEVICE = "cuda"

TEXT_PROMPTS = [
    "The most surprising thing about the ocean is",
    "In 1969, the first humans landed on the Moon. The mission",
    "A good way to learn a new language is to",
    "The city had been quiet for years, until one morning",
    "Photosynthesis is the process by which plants",
    "The recipe calls for three eggs, a cup of flour, and",
    "Once upon a time, in a village at the edge of a forest,",
    "The main difference between weather and climate is",
    "She opened the old letter and began to read:",
    "The invention of the printing press changed",
    "To fix a flat bicycle tire, first",
    "The history of mathematics begins with",
    "The detective looked at the room and immediately noticed",
    "Water boils at a lower temperature at high altitude because",
    "The team gathered around the whiteboard to plan",
    "Every winter, the birds fly south because",
]

CODE_PROMPTS = [
    "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
    "def reverse_string(s):\n    \"\"\"Return s reversed.\"\"\"\n",
    "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "def count_vowels(text):\n    \"\"\"Count vowels in text.\"\"\"\n",
    "def max_of_list(xs):\n    \"\"\"Return the largest element of xs.\"\"\"\n",
    "def factorial(n):\n    \"\"\"Return n! computed iteratively.\"\"\"\n",
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists.\"\"\"\n",
    "def word_count(sentence):\n    \"\"\"Return number of words.\"\"\"\n",
    "def celsius_to_fahrenheit(c):\n    \"\"\"Convert Celsius to F.\"\"\"\n",
    "def flatten(nested):\n    \"\"\"Flatten a list of lists.\"\"\"\n",
    "def gcd(a, b):\n    \"\"\"Greatest common divisor.\"\"\"\n",
    "def remove_duplicates(xs):\n    \"\"\"Drop duplicates, keep order.\"\"\"\n",
    "def binary_search(xs, target):\n    \"\"\"Index of target or -1.\"\"\"\n",
    "def sum_digits(n):\n    \"\"\"Sum of decimal digits of n.\"\"\"\n",
    "def title_case(s):\n    \"\"\"Capitalize each word of s.\"\"\"\n",
]


@torch.no_grad()
def gen_entropy(model, tok, prompts, ctx, max_new, temperature=0.8,
                top_k=50, seed=0):
    """Sample completions, return mean full-vocab entropy over generated
    tokens (through first EOS) — identical definition to
    grpo_code_stilt.seq_logprobs."""
    torch.manual_seed(seed)
    enc = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]
    Lp = max(len(e) for e in enc)
    B = len(enc)
    x = torch.full((B, Lp), tok.eos_token_id, dtype=torch.long)
    for i, e in enumerate(enc):
        x[i, Lp - len(e):] = e                       # left-pad
    x = x.to(DEVICE)
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, -ctx:])
        lg = logits[:, -1].float() / temperature
        v, ix = torch.topk(lg, top_k, dim=-1)
        probs = torch.softmax(v, -1)
        nxt = ix.gather(-1, torch.multinomial(probs, 1))
        nxt[finished] = tok.eos_token_id
        x = torch.cat([x, nxt], 1)
        finished |= nxt.squeeze(1) == tok.eos_token_id
        if finished.all():
            break

    # teacher-forced entropy pass, same mask logic as seq_logprobs
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x[:, -ctx:] if x.shape[1] > ctx else x)
    lp = torch.log_softmax(logits.float()[:, :-1], dim=-1)
    tgt = x[:, 1:]
    ent = -(lp.exp() * lp).sum(-1)                   # (B, L-1)
    gen_mask = torch.zeros_like(ent, dtype=torch.bool)
    gen_mask[:, Lp - 1:] = True
    pad_mask = tgt != tok.eos_token_id
    first_eos = (~pad_mask & gen_mask).float().cumsum(1) <= 1
    m = gen_mask & (pad_mask | ((tgt == tok.eos_token_id) & first_eos))
    n = m.sum().clamp_min(1).float()
    return (ent * m).sum().item() / n.item(), int(m.sum().item())


def load_model(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    sd = blob.get("model", blob.get("model_state", None))
    model.load_state_dict(sd)
    model.eval()
    return model, cfg, blob.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                    help="label=path; repeat, in ladder order")
    ap.add_argument("--init-from", default=None,
                    help="ckpt whose cfg seeds a random-init point")
    ap.add_argument("--arm", required=True, help="stj | sdpa (for labeling)")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompts-json", default=None,
                    help='optional {"text": [...], "code": [...]} override')
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt_sets = {"text": TEXT_PROMPTS, "code": CODE_PROMPTS}
    if args.prompts_json:
        with open(args.prompts_json) as f:
            prompt_sets = json.load(f)

    ladder = []
    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu",
                          weights_only=False)
        cfg = SimpleNamespace(**blob["args"])
        torch.manual_seed(args.seed)
        model = GPT(cfg).to(DEVICE).eval()
        ladder.append(("init", model, cfg))
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        model, cfg, step = load_model(path)
        ladder.append((label, model, cfg))

    rows = []
    for idx, (label, model, cfg) in enumerate(ladder):
        row = {"arm": args.arm, "stage_idx": idx, "stage": label}
        for pset, prompts in prompt_sets.items():
            e, ntok = gen_entropy(model, tok, prompts, cfg.ctx,
                                  args.max_new, seed=args.seed)
            row[f"ent_{pset}"] = e
            row[f"ntok_{pset}"] = ntok
        rows.append(row)
        print("  " + json.dumps(row))
        del model
        torch.cuda.empty_cache()

    out = args.out or f"results/gen_entropy_{args.arm}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"arm": args.arm, "seed": args.seed,
                   "max_new": args.max_new, "rows": rows}, f, indent=2)
    print(f"WROTE {out}")

    if args.wandb:
        import wandb
        run = wandb.init(entity="gulkoa", project="stieltjes-flash-attn",
                         name=f"gen-entropy-trajectory-{args.arm}",
                         config=vars(args))
        cols = sorted({k for r in rows for k in r})
        table = wandb.Table(columns=cols,
                            data=[[r.get(c) for c in cols] for r in rows])
        run.log({"gen_entropy_trajectory": table})
        for r in rows:
            run.log({f"ent_{k[4:]}": v for k, v in r.items()
                     if k.startswith("ent_")} | {"stage_idx": r["stage_idx"]})
        run.finish()


if __name__ == "__main__":
    main()
