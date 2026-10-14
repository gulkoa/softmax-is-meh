---
license: mit
tags:
  - attention
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# Stieltjes Attention LM

Small research language model (~8.5M parameters) with a custom attention
mechanism, trained on a synthetic multi-query associative-recall task:
the context interleaves key–value pairs with distractors, and the model
retrieves the values for queried keys far beyond its training length.

Tokens are space-separated integers. `0 k1 k2 1 v1 v2` binds a key to a
value; `2 k1 k2` queries it; `<bos>`/`<sep>`/`<eos>` wrap the sequence.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stieltjes-attention-mqmtar"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = "<bos> 0 10 20 1 111 222 0 0 0 30 40 1 55 66 2 10 20 2 30 40 <sep>"
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=13, do_sample=False)
print(tok.decode(out[0, ids.input_ids.shape[1]:]))
# -> 2 111 222 2 55 66 ...
```

Runs on CPU. A dedicated Inference Endpoint deploys directly via the
included `handler.py`.

Research artifact — no safety tuning.
