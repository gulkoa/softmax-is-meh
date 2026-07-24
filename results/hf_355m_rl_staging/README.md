---
license: mit
tags:
  - text-generation
  - conversational
  - reinforcement-learning
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1-355m-code-rl (preview)

**RL complete.** This repo serves the held-out-selected best
checkpoint (GRPO step 300 of 400): MBPP-test pass@1 4.1% /
pass@10 16.3%, vs 3.2% / 15.2% for its pre-RL base — a +27%
relative pass@1 gain from execution-reward RL at 355M scale.
Selection is by held-out eval across checkpoints, never by
training curve.

The 355M research language model
with a custom attention mechanism, RL-tuned for Python code synthesis
with execution-based rewards (GRPO: group-relative advantages,
KL-anchored to its SFT reference). It reasons in a visible
`<think>...</think>` block before writing a fenced code block.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1-355m-code-rl"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = ("<|user|>\nWrite a function to reverse a string.\n"
          "Your code should pass these tests:\n"
          "assert rev('ab') == 'ba'\n<|assistant|>\n")
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, ids.input_ids.shape[1]:]))
```

No hard token bounds. Lineage: stilt.1-355m → chat SFT → code SFT →
reasoning-format SFT → execution-reward GRPO.

## Family

| model | params | notes |
|---|---|---|
| [stilt.1-124m](https://huggingface.co/GulkoA/stilt.1-124m) | 124M | base |
| [stilt.1-124m-it](https://huggingface.co/GulkoA/stilt.1-124m-it) | 124M | chat |
| [stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m) | 355M | base |
| [stilt.1-355m-it](https://huggingface.co/GulkoA/stilt.1-355m-it) | 355M | chat |
| stilt.1-355m-code-rl | 355M | this repo — code RL, preview |

Research artifact — no safety tuning. Expect rough edges: this is a
small experimental model mid-way through RL training.
