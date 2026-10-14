"""Passkey retrieval at 1x-8x the training context for the SHIPPED models
(HF staging dirs / Hub repos, trust_remote_code), PI-paper protocol: a
5-digit key stated once at a given depth inside filler prose, padded to L
tokens, greedy completion of "The pass key is", exact match. Options: log-n
attention sharpening (--alpha, applied by scaling the score einsum of the
custom modeling code) and, for RoPE models, NTK theta scaling (--ntk: theta x
max(1, 2L/1024), the family's best setting). Learned-position models are
scored at 1024 only (they cannot extend).

  python eval_passkey_hf.py <model dir> [--lens 1024,2048,4096,8192] [--depths 0.1,0.5,0.9]
                            [--trials 20] [--alpha 0,0.35] [--ntk] [--seed 0]
"""
import argparse, math, os, random
import torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
_BOOST = 1.0
_orig_einsum = torch.einsum


def _einsum_boosted(eq, *ops, **kw):
    out = _orig_einsum(eq, *ops, **kw)
    return out * _BOOST if eq == "bhsd,bhtd->bhst" else out


torch.einsum = _einsum_boosted      # the custom Attn scores q.k through this einsum


def build(tok, L, depth, key):
    key_s = f"The pass key is {key}. Remember it. {key} is the pass key. "
    q = "What is the pass key? The pass key is"
    f_ids = tok(FILLER, add_special_tokens=False).input_ids
    k_ids = tok(key_s, add_special_tokens=False).input_ids
    q_ids = tok(q, add_special_tokens=False).input_ids
    room = L - len(k_ids) - len(q_ids)
    n_before = int(room * depth)
    before, after = [], []
    while len(before) < n_before:
        before += f_ids
    while len(after) < room - n_before:
        after += f_ids
    return before[:n_before] + k_ids + after[:room - n_before] + q_ids


def main():
    global _BOOST
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default="1024,2048,4096,8192")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--alpha", default="0,0.35")
    ap.add_argument("--ntk", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-len", type=int, default=1024)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).to(DEVICE).eval()
    cfg = model.config
    pos = "rope" if getattr(cfg, "rope", False) else ("nope" if getattr(cfg, "nope", False) else ("alibi" if getattr(cfg, "alibi", False) else "learned"))
    lens = [int(x) for x in a.lens.split(",")]
    if pos == "learned":
        lens = [L for L in lens if L <= cfg.ctx]
    depths = [float(x) for x in a.depths.split(",")]
    alphas = [float(x) for x in a.alpha.split(",")]
    attns = [m for m in model.modules() if hasattr(m, "rope_theta")]
    base_theta = float(getattr(cfg, "rope_theta", 10000.0))
    rng = random.Random(a.seed)
    keys = [rng.randint(10000, 99999) for _ in range(a.trials)]
    print(f"model {os.path.basename(a.model.rstrip('/'))} | positions {pos} | attn {getattr(cfg, 'attn', 'stieltjes')} | "
          f"ntk {'on' if a.ntk and pos == 'rope' else 'off'} | {a.trials} trials", flush=True)
    print(f"{'L':>6} {'alpha':>6} " + " ".join(f"d={d:g}".rjust(8) for d in depths) + "   mean", flush=True)
    for L in lens:
        if pos == "rope" and a.ntk:
            for m in attns:
                m.rope_theta = base_theta * max(1.0, 2.0 * L / a.train_len)
        for al in alphas:
            _BOOST = 1.0 + al * math.log(max(L, a.train_len) / a.train_len)
            accs = []
            for d in depths:
                hit = 0
                for key in keys:
                    ids = torch.tensor(build(tok, L, d, key), dtype=torch.long, device=DEVICE)[None]
                    with torch.no_grad():
                        out = model.generate(ids, max_new_tokens=6, do_sample=False, use_cache=True,
                                             pad_token_id=tok.eos_token_id)
                    hit += str(key) in tok.decode(out[0, ids.shape[1]:])
                accs.append(hit / len(keys))
            print(f"{L:>6} {al:>6g} " + " ".join(f"{x:>8.2f}" for x in accs) + f"   {sum(accs)/len(accs):.2f}", flush=True)
            _BOOST = 1.0
    for m in attns:
        m.rope_theta = base_theta
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
