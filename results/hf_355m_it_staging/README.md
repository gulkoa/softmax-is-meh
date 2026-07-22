---
license: mit
tags:
  - text-generation
  - conversational
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# stilt.1-355m-it

Instruction-tuned variant of
[stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m) — a 355M
research language model with a custom attention mechanism, pretrained
on a web+math+code mix. Fine-tuned on smol-smoltalk with a plain-text
chat template.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1-355m-it"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=60, do_sample=False)
print(tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True))
```

The chat template is plain text: `<|user|>\n{message}\n<|assistant|>\n`,
turns separated by the GPT-2 end-of-text token. No hard token bounds.

## Family

| model | params | notes |
|---|---|---|
| [stilt.1-124m](https://huggingface.co/GulkoA/stilt.1-124m) | 124M | base |
| [stilt.1-124m-it](https://huggingface.co/GulkoA/stilt.1-124m-it) | 124M | chat |
| [stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m) | 355M | base |
| stilt.1-355m-it | 355M | this repo |

## Endpoint (OpenAI-compatible bodies)

Deployed as an Inference Endpoint (Default container, `handler.py`),
the model accepts OpenAI chat-completion request bodies at the endpoint
URL and returns OpenAI-shaped responses:

```bash
curl <ENDPOINT_URL> -H "Authorization: Bearer <HF_TOKEN>" \
  -H "Content-Type: application/json" -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 60, "temperature": 0.7}'
```

Research artifact — no safety tuning, no content filtering.
