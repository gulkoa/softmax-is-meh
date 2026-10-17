---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1.1-355m-nope-qk-softmax-baseline-seed1

**Stilt** is a family of research language models built on a custom
attention mechanism. The name honors T.J. Stieltjes — and the
black-winged stilt, a bird whose legs are as heavy-tailed as our
attention weights.

`stilt.1.1-355m-nope-qk-softmax-baseline-seed1`: a second-seed replicate of
`stilt.1.1-355m-nope-qk-softmax-baseline` and the softmax control twin of
`stilt.1.1-355m-nope-qk-seed1` — identical data (15B tokens, 70/15/15 FineWeb-Edu / FineMath /
Python), schedule, seed (1), **NoPE** and **QK-norm**, with standard softmax
attention in place of Stieltjes. 355M parameters, GPT-2-medium-style (24
layers, GPT-2 BPE), lr 6e-4, final val ppl 11.87 (seed 0: 11.90).
Trained at context 1024; no architectural sequence cap.

This is a base (completion-style) model — no instruction tuning.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1.1-355m-nope-qk-softmax-baseline-seed1"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```
