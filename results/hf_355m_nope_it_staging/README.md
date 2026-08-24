---
license: mit
tags:
  - text-generation
  - research
  - custom_code
  - conversational
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1.1-355m-nope-it

**Stilt** is a family of research language models built on a custom
attention mechanism. The name honors T.J. Stieltjes — and the
black-winged stilt, a bird whose legs are as heavy-tailed as our
attention weights.

`stilt.1.1-355m-nope-it`: the instruction-tuned chat variant of
[`GulkoA/stilt.1.1-355m-nope`](https://huggingface.co/GulkoA/stilt.1.1-355m-nope)
(355M, GPT-2-medium-style, **no positional embeddings**). SFT on full
smoltalk, conversations kept whole at context 2048, plus a synthetic
deep-recall slice: dialogues where a fact stated early must be
retrieved 600–1800 tokens later. On the deep-context recall probe this
lifts recall at depth 1200 from 0.00 (SFT without the slice) to 0.83 —
retrieval demand in the data unlocks what the position-free
architecture already permits, with a sharp boundary at the trained
depth envelope.

No architectural sequence cap; KV-cached generation; no hard token
bounds.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1.1-355m-nope-it"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

chat = "<|user|>What is the capital of France?<|assistant|>"
ids = tok(chat, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=60, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```

Research model: small, honest about its size. Expect confident
nonsense outside its competence.
