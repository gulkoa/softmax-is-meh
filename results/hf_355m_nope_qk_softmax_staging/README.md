---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1.1-355m-nope-qk-softmax-baseline

**Stilt** is a family of research language models built on a custom
attention mechanism. The name honors T.J. Stieltjes — and the
black-winged stilt, a bird whose legs are as heavy-tailed as our
attention weights.

`stilt.1.1-355m-nope-qk-softmax-baseline`: the control twin of
`stilt.1.1-355m-nope-qk` — identical data (15B tokens, 70/15/15
FineWeb-Edu / FineMath / Python), schedule, **NoPE** and **QK-norm**, with
standard softmax attention in place of Stieltjes. 355M parameters,
GPT-2-medium-style (24 layers, GPT-2 BPE), lr 6e-4, final val ppl 11.90.
QK-norm is what lets softmax train at this learning rate: without it the
softmax NoPE arm needed lr 2e-4 (`stilt.1.1-355m-nope-softmax-baseline`,
12.82). Trained at context 1024; no architectural sequence cap. Length
extrapolation (PG-19 deep-tail perplexity with log-n attention
sharpening): 29.0 / 59.1 / 169.9 at 2× / 4× / 8× the training context.

This is a base (completion-style) model — no instruction tuning.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1.1-355m-nope-qk-softmax-baseline"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```
