---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1-355m-softmax-baseline

Softmax-attention control twin of
[stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m): identical
architecture (GPT-2-medium-style, 24 layers, ctx 1024, GPT-2 BPE),
identical 15B-token web+math+code data schedule — with standard softmax
attention in place of the custom mechanism. Published for controlled
comparisons; if you just want the best model of the family, use
stilt.1-355m or its `-it` variant.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1-355m-softmax-baseline"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```

Runs on CPU or GPU. No hard token bounds.

## Family

| model | params | notes |
|---|---|---|
| [stilt.1-124m](https://huggingface.co/GulkoA/stilt.1-124m) | 124M | base |
| [stilt.1-124m-it](https://huggingface.co/GulkoA/stilt.1-124m-it) | 124M | chat |
| [stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m) | 355M | base |
| stilt.1-355m-softmax-baseline | 355M | this repo — control twin |

Research artifact — no instruction tuning, no safety tuning.
