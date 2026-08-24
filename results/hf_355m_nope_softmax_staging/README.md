---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1.1-355m-nope-softmax-baseline

Control twin for
[`GulkoA/stilt.1.1-355m-nope`](https://huggingface.co/GulkoA/stilt.1.1-355m-nope):
identical 355M GPT-2-medium-style architecture, identical 15B-token
70/15/15 FineWeb-Edu/FineMath/code mix, identical NoPE setup (no
positional embeddings) — but standard softmax attention instead of
Stieltjes.

Published for reproducibility of the head-to-head: this arm required a
3× smaller learning rate to train at all (diverged at 6e-4 and 3e-4,
clean at 2e-4) and finished at val ppl 12.82 vs the Stieltjes twin's
12.02 at 6e-4. Base (completion-style) model, no instruction tuning.
No architectural sequence cap; KV-cached generation; no hard token
bounds.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1.1-355m-nope-softmax-baseline"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```
