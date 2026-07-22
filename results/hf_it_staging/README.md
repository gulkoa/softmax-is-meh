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

# stilt.1-124m-it

Instruction-tuned variant of
[stilt.1-124m](https://huggingface.co/GulkoA/stilt.1-124m) — a 124M
research language model with a custom attention mechanism. Fine-tuned
on smol-smoltalk with a plain-text chat template.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/stilt.1-124m-it"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=60, do_sample=False)
print(tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True))
```

The chat template is plain text: `<|user|>\n{message}\n<|assistant|>\n`,
turns separated by the GPT-2 end-of-text token.

## Family

| model | params | notes |
|---|---|---|
| [stilt.1-124m](https://huggingface.co/GulkoA/stilt.1-124m) | 124M | base |
| stilt.1-124m-it | 124M | this repo |
| stilt.1-355m | 355M | [GulkoA/stilt.1-355m](https://huggingface.co/GulkoA/stilt.1-355m) |

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

(The classic HF `{"inputs": ..., "parameters": ...}` format also still
works. Note: the OpenAI *SDK* appends `/v1/chat/completions` to its
base URL, which the handler container does not route — use plain HTTP
as above, or any client that posts to the URL you give it.)

**Token limits**: none enforced — the tokenizer does not truncate and
generation length is caller-controlled. The model was trained at
context 1024; beyond that, positions saturate and it keeps generating
with gradually degrading quality (attention itself has no length
limit).

Research artifact — small model, minimal alignment; expect
124M-class limitations.
