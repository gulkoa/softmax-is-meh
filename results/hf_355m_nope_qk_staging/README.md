---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1.1-355m-nope-qk

**Stilt** is a family of research language models built on a custom
attention mechanism. The name honors T.J. Stieltjes — and the
black-winged stilt, a bird whose legs are as heavy-tailed as our
attention weights.

`stilt.1.1-355m-nope-qk`: 355M parameters, GPT-2-medium-style (24 layers,
GPT-2 BPE), trained on 15B tokens of a 70/15/15 mix of FineWeb-Edu,
FineMath, and Python code — **no positional embeddings (NoPE)** and
**QK-norm** (RMS-normalized queries and keys with a per-head learnable
gain) on top of Stieltjes attention (q=4), lr 6e-4. Final val ppl 11.93.
It is the Stieltjes arm of a matched-recipe pair: its softmax twin
`stilt.1.1-355m-nope-qk-softmax-baseline` reaches 11.90 with the identical
recipe, and its no-norm sibling `stilt.1.1-355m-nope` reaches 12.02.
Trained at context 1024; NoPE has no architectural sequence cap. Length
extrapolation (PG-19 deep-tail perplexity with log-n attention
sharpening): 28.6 / 52.8 / 152.7 at 2× / 4× / 8× the training context.

This is a base (completion-style) model — no instruction tuning.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1.1-355m-nope-qk"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```
