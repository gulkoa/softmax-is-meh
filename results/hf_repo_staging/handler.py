"""Custom handler for Hugging Face Inference Endpoints.

Deploying this repo as a dedicated Inference Endpoint (any CPU tier is
sufficient for the 8.5M model) picks this up automatically:
input  {"inputs": "<bos> 0 10 20 1 111 222 ... 2 10 20 <sep>",
        "parameters": {"max_new_tokens": 13}}
output [{"generated_text": "..."}]
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class EndpointHandler:
    def __init__(self, path=""):
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True)
        self.model.eval()

    def __call__(self, data):
        prompt = data.get("inputs", "")
        params = data.get("parameters", {}) or {}
        max_new = int(params.get("max_new_tokens", 13))
        ids = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new,
                                      do_sample=False)
        gen = self.tokenizer.decode(out[0, ids.input_ids.shape[1]:],
                                    skip_special_tokens=False)
        for s, i in [("<pad>", 256), ("<bos>", 257), ("<sep>", 258),
                     ("<eos>", 259)]:
            gen = gen.replace(s, str(i))
        return [{"generated_text": gen.strip()}]
