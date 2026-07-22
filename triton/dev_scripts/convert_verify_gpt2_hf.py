"""M4: convert the trained Stieltjes GPT-2 checkpoint to the HF staging
repo and verify — (A) logits parity vs the trainer model (dense stieltjes
vs the fused-kernel trainer path), (B) cached vs uncached generation
equivalence, (C) a sample completion. Saves safetensors + config +
GPT-2 tokenizer files into the staging dir on success."""
import json
import os
import sys
from types import SimpleNamespace

import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

STAGING = (sys.argv[2] if len(sys.argv) > 2 else
           "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_gpt2_staging")
CKPT = (sys.argv[1] if len(sys.argv) > 1 else
        "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4_s0.pt")
sys.path.insert(0, STAGING)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts")

from transformers import AutoTokenizer  # noqa: E402
import configuration_stieltjes_gpt2 as C  # noqa: E402
sys.modules["configuration_stieltjes_gpt2"] = C
src = open(f"{STAGING}/modeling_stieltjes_gpt2.py").read().replace(
    "from .configuration_stieltjes_gpt2", "from configuration_stieltjes_gpt2")
ns = {}
exec(compile(src, "modeling_stieltjes_gpt2.py", "exec"), ns)
StieltjesGPT2Config = C.StieltjesGPT2Config
StieltjesGPT2ForCausalLM = ns["StieltjesGPT2ForCausalLM"]

from train_gpt2_stieltjes import GPT  # noqa: E402

DEVICE = torch.device("cuda")

blob = torch.load(CKPT, map_location="cpu", weights_only=False)
targs = SimpleNamespace(**blob["args"])
assert targs.attn in ("stj", "sdpa")

cfg = StieltjesGPT2Config(n_layer=targs.n_layer, n_head=targs.n_head,
                          n_embd=targs.n_embd, ctx=targs.ctx,
                          stj_q=targs.stj_q,
                          attn=("sdpa" if targs.attn == "sdpa"
                                else "stieltjes"))
hf = StieltjesGPT2ForCausalLM(cfg)
res = hf.load_state_dict(blob["model"], strict=False)
print("missing:", res.missing_keys, "| unexpected:", res.unexpected_keys)
assert not res.unexpected_keys and not res.missing_keys
hf = hf.to(DEVICE).eval()

trainer = GPT(targs).to(DEVICE)
trainer.load_state_dict(blob["model"])
trainer.eval()

tok = AutoTokenizer.from_pretrained("gpt2")
prompt = "The history of mathematics begins with"
ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

# A: logits parity (dense fp32 serving path vs fused bf16 trainer path)
with torch.no_grad():
    lg_hf = hf(input_ids=ids, use_cache=False).logits
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg_tr, _ = trainer(ids)
diff = (lg_hf - lg_tr.float()).abs().max().item()
agree = (lg_hf[0, -1].argmax() == lg_tr[0, -1].float().argmax()).item()
print(f"A: max|logit diff| {diff:.3f} (bf16 kernel vs fp32 dense), "
      f"next-token argmax agree: {agree}")
assert agree

# B: cached vs uncached greedy generation equivalence
with torch.no_grad():
    out_cache = hf.generate(ids, max_new_tokens=30, do_sample=False)
    hf.config.use_cache = False
    cur = ids
    for _ in range(30):
        lg = hf(input_ids=cur, use_cache=False).logits
        cur = torch.cat([cur, lg[0, -1].argmax()[None, None]], 1)
    hf.config.use_cache = True
same = torch.equal(out_cache, cur)
print("B: cached == uncached greedy:", same)
assert same

# C: sample
print("C:", repr(tok.decode(out_cache[0])))

# save
cfg.auto_map = {
    "AutoConfig":
        "configuration_stieltjes_gpt2.StieltjesGPT2Config",
    "AutoModelForCausalLM":
        "modeling_stieltjes_gpt2.StieltjesGPT2ForCausalLM",
}
hf.save_pretrained(STAGING, safe_serialization=True)
tok.save_pretrained(STAGING)
tc = json.load(open(f"{STAGING}/tokenizer_config.json"))
tc["model_max_length"] = 1000000000000  # policy: no hard token bounds
json.dump(tc, open(f"{STAGING}/tokenizer_config.json", "w"), indent=2)
gcp = f"{STAGING}/generation_config.json"
if os.path.exists(gcp):
    gc = json.load(open(gcp))
    gc["max_length"] = None
    json.dump(gc, open(gcp, "w"), indent=2)
print("SAVED to", STAGING)
print("VERIFIED")
