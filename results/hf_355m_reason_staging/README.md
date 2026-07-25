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

# stilt.1-355m-reason-rl

The strongest model of the Stilt family: the 355M research language
model with a custom attention mechanism, RL-tuned for multi-step
problem solving (math word problems, logic, structured reasoning)
with verified-answer rewards plus an LLM judge grading its visible
`<think>` traces. Held-out-selected checkpoint (GRPO step 300/400):

| held-out channel | before RL | after RL |
|---|---|---|
| GSM8K (test split) acc@1 | 3.0% | **6.4%** (+113%) |
| ARC-Easy (test) acc@1 | 32.5% | **58.9%** |
| fresh logic tasks acc@1 | 5.5% | **10.0%** |

It thinks in `<think>...</think>`, then answers after `Answer:`.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1-355m-reason-rl"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = ("<|user|>\nWhat is 17 + 25?\nThink step by step inside "
          "<think> and </think>, then give your final answer on a new "
          "line starting with 'Answer:'.\n<|assistant|>\n")
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, ids.input_ids.shape[1]:]))
```

No hard token bounds. Lineage: stilt.1-355m → chat SFT → code SFT →
reasoning-format SFT → verified+judge GRPO (local Qwen judge, zero
API spend).

## Family

| model | params | notes |
|---|---|---|
| [stilt.1-124m-it](https://huggingface.co/GulkoA/stilt.1-124m-it) | 124M | chat |
| [stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m) | 355M | base |
| [stilt.1-355m-it](https://huggingface.co/GulkoA/stilt.1-355m-it) | 355M | chat |
| [stilt.1-355m-code-rl](https://huggingface.co/GulkoA/stilt.1-355m-code-rl) | 355M | code RL |
| stilt.1-355m-reason-rl | 355M | this repo — reasoning RL |

Research artifact — no safety tuning. A small experimental model:
verify anything important.
