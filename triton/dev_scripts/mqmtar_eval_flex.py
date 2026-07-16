"""
Long-context (16k-65k) eval of MQMTAR softmax+NAPE checkpoints via
FlexAttention (memory-linear ALiBi softmax — dense eval dies at 16k).

The trained weights (qkv/proj/mlp/ln/emb) are attention-implementation-
agnostic: we rebuild the model with a flex_attention forward computing
EXACTLY the softmaxd math — softmax((q k^T) / sqrt(d) + slope*(kv-q)) with
NAPE slopes [1/1..1/8, 0x8] — and load the state_dict unchanged.

Equivalence is asserted on a short prompt against the dense path before
long evals run.

Usage: python mqmtar_eval_flex.py <softmaxd-ckpt.pt> --lengths 16384 32768 65536
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mqmtar_headtohead_v8 as M  # noqa: E402  (import only)

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

SPLIT_INDEX = {64: 0, 128: 1, 256: 2, 512: 3, 1024: 4, 2048: 5, 4096: 6,
               8192: 7, 16384: 8, 32768: 9, 65536: 10}

_flex = torch.compile(flex_attention, dynamic=False)


class FlexNapeAttn(nn.Module):
    """softmaxd --nape forward via flex_attention (same parameters).

    block_mask is set externally per split (set_block_mask) so S stays
    CONSTANT across the 13 generation steps — otherwise torch.compile
    recompiles/bails to the eager math path, which materializes dense
    scores (111 GiB OOM at 16k, job 12380048)."""

    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.n_head, cfg.head_dim
        inner = self.h * self.hd
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, cfg.d_model, bias=False)
        slopes = torch.cat([1.0 / torch.arange(1, self.h // 2 + 1),
                            torch.zeros(self.h - self.h // 2)])
        self.register_buffer("slopes", slopes, persistent=False)
        self.block_mask = None

    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).split(self.h * self.hd, dim=2)
        q = q.view(B, S, self.h, self.hd).transpose(1, 2).contiguous()
        k = k.view(B, S, self.h, self.hd).transpose(1, 2).contiguous()
        v = v.view(B, S, self.h, self.hd).transpose(1, 2).contiguous()
        slopes = self.slopes

        def score_mod(score, b, h, qi, ki):
            return score + slopes[h] * (ki - qi).to(score.dtype)

        o = _flex(q, k, v, score_mod=score_mod, block_mask=self.block_mask,
                  scale=1.0 / (self.hd ** 0.5))
        return self.proj(o.transpose(1, 2).reshape(B, S, self.h * self.hd))


def set_block_masks(model, S, device):
    mask = create_block_mask(lambda b, h, qi, ki: qi >= ki,
                             None, None, S, S, device=device)
    for blk in model.blocks:
        blk.attn.block_mask = mask


@torch.no_grad()
def eval_split_fixed_len(model, src_list, trg_list, device, n, bs):
    """Greedy exact-match eval with CONSTANT sequence length: x is
    pre-allocated at prompt_len + TRG_LEN and filled in place, so flex
    compiles once per split (trailing PADs are causally inert)."""
    correct = 0
    n = min(n, len(src_list))
    for lo in range(0, n, bs):
        idxs = range(lo, min(lo + bs, n))
        prompts = [np.concatenate(([M.BOS], src_list[i], [M.SEP]))
                   for i in idxs]
        Lp = max(len(p) for p in prompts)
        x = np.full((len(prompts), Lp + M.TRG_LEN), M.PAD, dtype=np.int64)
        for j, p in enumerate(prompts):
            x[j, Lp - len(p):Lp] = p           # LEFT-pad
        x = torch.from_numpy(x).to(device)
        set_block_masks(model, x.shape[1], device)
        for t in range(M.TRG_LEN):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            x[:, Lp + t] = logits[:, Lp + t - 1].argmax(-1)
        gen = x[:, Lp:Lp + M.TRG_LEN].cpu().numpy()
        for j, i in enumerate(idxs):
            if np.array_equal(gen[j], trg_list[i][:M.TRG_LEN]):
                correct += 1
    return correct / n


def build_flex_model(cfg, state_dict, device):
    model = M.MQMTARModel(cfg)
    for blk in model.blocks:
        flex = FlexNapeAttn(cfg)
        blk.attn = flex
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[16384, 32768, 65536])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--data", default=os.environ.get(
        "MQMTAR_DATA",
        "/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/"
        "50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4"))
    args = ap.parse_args()
    device = torch.device("cuda")

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    assert cfg.arm == "softmaxd" and cfg.nape, "flex eval is softmaxd+nape only"
    label = getattr(cfg, "arm_label", "softmaxd-nape")

    dense = M.MQMTARModel(cfg).to(device)
    dense.load_state_dict(blob["state_dict"])
    dense.eval()
    flexm = build_flex_model(cfg, blob["state_dict"], device)
    flexm.eval()

    # equivalence gate: dense/old loop vs flex/fixed-len loop must agree
    s64 = M.read_split(os.path.join(args.data, "test_0_64"))
    acc_d = M.eval_split(dense, s64[0][:50], s64[1][:50], device, 50, 25)
    acc_f = eval_split_fixed_len(flexm, s64[0][:50], s64[1][:50], device,
                                 50, 25)
    print(f"equivalence gate @64 (n=50): dense {acc_d:.3f} flex {acc_f:.3f}",
          flush=True)
    assert abs(acc_d - acc_f) <= 0.04, "dense/flex disagree — abort"
    del dense
    torch.cuda.empty_cache()

    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"mqmtar-flex-{label}-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config={**blob["args"], "eval": "flex-attention long-context",
                "recovered_from": args.ckpt},
    )
    for L in args.lengths:
        s, t = M.read_split(
            os.path.join(args.data, f"test_{SPLIT_INDEX[L]}_{L}"))
        bs = max(1, min(16, 2 ** 21 // L))
        t0 = time.time()
        acc = eval_split_fixed_len(flexm, s, t, device, args.n, bs)
        print(f"  len {L:6d}: acc {acc:.4f}  (n={args.n}, "
              f"{time.time()-t0:.0f}s)", flush=True)
        run.log({f"final/acc_{L}": acc})
        run.summary[f"acc_{L}"] = acc
    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
