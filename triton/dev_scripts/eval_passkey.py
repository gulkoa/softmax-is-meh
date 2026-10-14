"""Passkey retrieval at 1x-8x the training context (Press-style / PI-paper
protocol) for BASE checkpoints: a 5-digit pass key is stated once at a given
depth inside filler prose, the context is padded to L tokens, and the model
greedily completes "The pass key is" — exact match on the number. Reports
accuracy per (L, depth) for the raw model and with log-n attention
sharpening (alpha), the same knob as eval_longctx_scaled.py.

  python eval_passkey.py <ckpt.pt> [--lens 1024,2048,4096,8192] [--depths 0.1,0.5,0.9]
                         [--trials 20] [--alpha 0,0.35] [--seed 0]
"""
import argparse, math, os, random, sys
from types import SimpleNamespace
import torch
import torch.nn.functional as F
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt2_stieltjes as T  # noqa: E402
from train_gpt2_stieltjes import GPT  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
_BOOST = 1.0
_orig_stj, _orig_sdpa = T.stieltjes_attention, F.scaled_dot_product_attention


def _stj_boosted(*a, **kw):
    if "sm_scale" in kw:
        kw["sm_scale"] = kw["sm_scale"] * _BOOST
    return _orig_stj(*a, **kw)


def _sdpa_boosted(q, k, v, **kw):
    kw.setdefault("scale", _BOOST / math.sqrt(q.shape[-1]))
    return _orig_sdpa(q, k, v, **kw)


T.stieltjes_attention = _stj_boosted
F.scaled_dot_product_attention = _sdpa_boosted


def build(tok, L, depth, key, template="standard"):
    """Token ids of length <= L: filler, the key sentence at `depth`, filler, question."""
    key_s = f"The pass key is {key}. Remember it. {key} is the pass key. "
    q = "What is the pass key? The pass key is" if template == "standard" else "The pass key is"
    f_ids = tok(FILLER, add_special_tokens=False).input_ids
    k_ids = tok(key_s, add_special_tokens=False).input_ids
    q_ids = tok(q, add_special_tokens=False).input_ids
    room = L - len(k_ids) - len(q_ids)
    n_before = int(room * depth)
    before, after = [], []
    while len(before) < n_before:
        before += f_ids
    before = before[:n_before]
    while len(after) < room - n_before:
        after += f_ids
    after = after[:room - n_before]
    return before + k_ids + after + q_ids


@torch.no_grad()
def greedy(model, ids, n_new):
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None]
    out = []
    for _ in range(n_new):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits, _ = model(x)
        nxt = logits[0, -1].argmax().item()
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], 1)
    return out


def main():
    global _BOOST
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--lens", default="1024,2048,4096,8192")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--alpha", default="0,0.35")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-len", type=int, default=1024)
    ap.add_argument("--stj-iters", type=int, default=0, dest="stj_iters",
                    help="override the Stieltjes Halley iterations (0 = the checkpoint's)")
    ap.add_argument("--template", default="standard", choices=["standard", "noquestion"],
                    help="noquestion: the context ends with the cue, no question repeating 'is the pass key'")
    ap.add_argument("--show-fails", type=int, default=0, dest="show_fails",
                    help="print up to N failed completions per (L, alpha, depth) cell")
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    if a.stj_iters:
        cfg.stj_iters = a.stj_iters      # longer contexts may need more λ-solve iterations
    model = GPT(cfg).to(DEVICE).eval()
    model.load_state_dict(blob["model"])
    tname = getattr(cfg, "tokenizer", "gpt2") or "gpt2"
    tok = AutoTokenizer.from_pretrained((getattr(cfg, "tokenizer_path", None) or "gpt2") if tname != "gpt2" else "gpt2")
    lens = [int(x) for x in a.lens.split(",")]
    depths = [float(x) for x in a.depths.split(",")]
    alphas = [float(x) for x in a.alpha.split(",")]
    rng = random.Random(a.seed)
    keys = [rng.randint(10000, 99999) for _ in range(a.trials)]
    print(f"ckpt {os.path.basename(a.ckpt)} | tokenizer {tname} | positions {'nope' if getattr(cfg,'nope',False) else ('rope' if getattr(cfg,'rope',False) else ('alibi' if getattr(cfg,'alibi',False) else 'learned'))} | template {a.template} | {a.trials} trials", flush=True)
    print(f"{'L':>6} {'alpha':>6} " + " ".join(f"d={d:g}".rjust(8) for d in depths) + "   mean", flush=True)
    for L in lens:
        for al in alphas:
            _BOOST = 1.0 + al * math.log(max(L, a.train_len) / a.train_len)
            accs, fails = [], []
            for d in depths:
                hit = 0
                for key in keys:
                    ids = build(tok, L, d, key, a.template)
                    gen = tok.decode(greedy(model, ids, 6))
                    ok = str(key) in gen
                    hit += ok
                    if not ok and sum(1 for f in fails if f[0] == d) < a.show_fails:
                        fails.append((d, key, gen))
                accs.append(hit / len(keys))
            _BOOST = 1.0
            print(f"{L:>6} {al:>6g} " + " ".join(f"{x:>8.2f}" for x in accs) + f"   {sum(accs)/len(accs):.2f}", flush=True)
            for d, key, gen in fails:
                print(f"         fail d={d:g} key={key} -> {gen!r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
