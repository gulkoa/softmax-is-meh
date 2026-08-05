---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1.1-355m-nope

**Stilt** is a family of research language models built on a custom
attention mechanism. The name honors T.J. Stieltjes — and the
black-winged stilt, a bird whose legs are as heavy-tailed as our
attention weights.

`stilt.1.1-355m-nope`: 355M parameters, GPT-2-medium-style (24 layers,
GPT-2 BPE), trained on 15B tokens of a 70/15/15 mix of FineWeb-Edu,
FineMath, and Python code — **with no positional embeddings** (NoPE).
The architecture has no positional lookup table, so nothing in the
model caps sequence length; it was trained at context 1024 and degrades
gracefully rather than erroring beyond it. KV-cached generation, no
hard token bounds.

This is a base (completion-style) model — no instruction tuning.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1.1-355m-nope"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```

Runs on CPU or GPU. A dedicated Inference Endpoint deploys directly via
the included `handler.py`.

## Family

| model | params | data | status |
|---|---|---|---|
| stilt.1-124m | 124M | FineWeb-Edu 10B | [GulkoA/stilt.1-124m](https://huggingface.co/GulkoA/stilt.1-124m) |
| stilt.1-124m-it | 124M | + chat SFT | [GulkoA/stilt.1-124m-it](https://huggingface.co/GulkoA/stilt.1-124m-it) |
| stilt.1-355m | 355M | web+math+code 15B | [GulkoA/stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m) |
| stilt.1-355m-it | 355M | + chat SFT | [GulkoA/stilt.1-355m-it](https://huggingface.co/GulkoA/stilt.1-355m-it) |
| stilt.1.1-355m-nope | 355M | web+math+code 15B, no positional embeddings | this repo |

Research artifacts — no instruction tuning, no safety tuning. `-it`
suffixes mark instruction-tuned variants.
