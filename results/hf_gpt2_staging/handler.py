"""Custom handler for Hugging Face Inference Endpoints.

input  {"inputs": "Once upon a time", "parameters": {"max_new_tokens": 60,
        "do_sample": true, "top_k": 40, "temperature": 1.0}}
output [{"generated_text": "..."}]
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class EndpointHandler:
    def __init__(self, path=""):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True)
        self.model.eval()

    def __call__(self, data):
        prompt = data.get("inputs", "")
        p = data.get("parameters", {}) or {}
        ids = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=int(p.get("max_new_tokens", 60)),
                do_sample=bool(p.get("do_sample", True)),
                top_k=int(p.get("top_k", 40)),
                temperature=float(p.get("temperature", 1.0)),
            )
        return [{"generated_text": self.tokenizer.decode(
            out[0], skip_special_tokens=True)}]
