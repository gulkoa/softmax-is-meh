"""Calibration reference (advisor request, 2026-08-25 meeting): run the
project's M3 battery — held-out FineWeb-Edu ppl (val shard, same loader
as training), WikiText-103 ppl, LAMBADA last-word accuracy — on STOCK
Hugging Face GPT-2 checkpoints with the exact same protocol as
eval_gpt2_stieltjes.py, so our numbers have an external anchor.

Usage: python eval_gpt2_hf_reference.py [hf_id ...]
       (default: openai-community/gpt2 openai-community/gpt2-medium)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_gpt2_stieltjes as E  # noqa: E402  (fineweb_ppl / wikitext_ppl / lambada_acc)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


class HFWrap(nn.Module):
    """Adapt a HF causal LM to the project's (logits, loss) = model(x, y)
    convention; loss = mean CE over ALL ctx positions against y (the
    already-shifted targets), exactly as the project GPT computes it."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, y=None):
        logits = self.m(input_ids=x).logits
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y.reshape(-1))
        return logits, loss


def main():
    ids = sys.argv[1:] or ["openai-community/gpt2", "openai-community/gpt2-medium"]
    tok = AutoTokenizer.from_pretrained("gpt2")
    for hf_id in ids:
        m = AutoModelForCausalLM.from_pretrained(hf_id).to(E.DEVICE).eval()
        model = HFWrap(m)
        ctx = min(1024, m.config.n_positions)
        nparam = sum(p.numel() for p in m.parameters())
        print(f"== {hf_id}: {nparam/1e6:.1f}M params, ctx {ctx}", flush=True)
        with torch.no_grad():
            print(f"fineweb-val ppl : {E.fineweb_ppl(model, ctx):8.2f}", flush=True)
            print(f"wikitext103 ppl : {E.wikitext_ppl(model, tok, ctx):8.2f}", flush=True)
            print(f"lambada acc     : {E.lambada_acc(model, tok):8.3f}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
