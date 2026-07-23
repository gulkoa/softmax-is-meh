"""Custom handler for Hugging Face Inference Endpoints.

Accepts BOTH formats:

1. OpenAI chat format (POST the same body you'd send to
   /v1/chat/completions):
     {"messages": [{"role": "user", "content": "Hi"}],
      "max_tokens": 60, "temperature": 0.7}
   -> OpenAI-shaped chat.completion response.

2. Classic HF format:
     {"inputs": "raw prompt", "parameters": {"max_new_tokens": 60, ...}}
   -> [{"generated_text": "..."}]
"""
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

U, A = "<|user|>\n", "<|assistant|>\n"


class EndpointHandler:
    def __init__(self, path=""):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True)
        self.model.eval()

    def _generate(self, prompt, max_new, temperature, top_p, do_sample,
                  repetition_penalty=1.15):
        ids = self.tokenizer(prompt, return_tensors="pt")
        n_prompt = ids.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new,
                do_sample=do_sample and temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0, n_prompt:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, n_prompt, int(gen_ids.shape[0])

    def _chat(self, data):
        prompt = ""
        for m in data.get("messages", []):
            role, content = m.get("role"), m.get("content", "")
            if role == "user":
                prompt += U + content + "\n"
            elif role == "assistant":
                prompt += A + content + self.tokenizer.eos_token
            elif role == "system":
                # no trained system channel; fold into the first user turn
                prompt += U + "(instructions) " + content + "\n"
        prompt += A
        text, n_in, n_out = self._generate(
            prompt,
            max_new=int(data.get("max_tokens") or 256),
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.95)),
            do_sample=True,
            repetition_penalty=float(data.get("repetition_penalty", 1.15)),
        )
        # trim at a spurious next-turn marker if the model emits one
        text = text.split("<|user|>")[0].strip()
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", "stilt.1-124m-it"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop" if n_out < int(
                    data.get("max_tokens") or 256) else "length",
            }],
            "usage": {"prompt_tokens": n_in,
                      "completion_tokens": n_out,
                      "total_tokens": n_in + n_out},
        }

    def __call__(self, data):
        if "messages" in data:
            return self._chat(data)
        p = data.get("parameters", {}) or {}
        text, _, _ = self._generate(
            data.get("inputs", ""),
            max_new=int(p.get("max_new_tokens", 60)),
            temperature=float(p.get("temperature", 1.0)),
            top_p=float(p.get("top_p", 1.0)),
            do_sample=bool(p.get("do_sample", True)),
        )
        return [{"generated_text": text}]
