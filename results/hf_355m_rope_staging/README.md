---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1.1-355m-rope

**Stilt** is a family of research language models built on a custom
attention mechanism. The name honors T.J. Stieltjes — and the
black-winged stilt, a bird whose legs are as heavy-tailed as our
attention weights.

`stilt.1.1-355m-rope`: 355M parameters, GPT-2-medium-style (24 layers,
GPT-2 BPE), trained on 15B tokens of a 70/15/15 mix of FineWeb-Edu,
FineMath, and Python code — **rotary position embeddings** (rotate-half
RoPE, θ=10000) instead of a learned position table. Final val ppl
11.68 — the strongest base model in the family at this scale (RoPE
11.68 < NoPE 12.02 < learned-positional 12.22 at matched budget).
Trained at context 1024; RoPE has no architectural sequence cap, and
the rotary geometry makes NTK/PI-style context extension applicable.

This is a base (completion-style) model — no instruction tuning.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1.1-355m-rope"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```
