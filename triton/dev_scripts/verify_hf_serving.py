"""Verify the HF trust_remote_code packaging of the Stieltjes LM:
AutoModelForCausalLM + AutoTokenizer -> generate() must reproduce the
native model's exact-match behavior on real MQMTAR samples (GPU)."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = sys.argv[1] if len(sys.argv) > 1 else (
    "/tmp/claude-41244/-users-PAS2402-alexg-softmax/"
    "89d1d46a-ca91-4553-b27b-596dc1e35dfb/scratchpad/hf_repo")
DATA = ("/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/"
        "3M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4")
N_SAMPLES = 25

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True)
model = model.cuda().eval()
print("loaded:", type(model).__name__, "on", next(model.parameters()).device)

srcs = open(f"{DATA}/test_0_64.src").readlines()[:N_SAMPLES]
trgs = open(f"{DATA}/test_0_64.trg").readlines()[:N_SAMPLES]
correct = 0
for i, (src, trg) in enumerate(zip(srcs, trgs)):
    ids = tok("<bos> " + src.strip() + " <sep>", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=13, do_sample=False)
    gen = tok.decode(out[0, ids.input_ids.shape[1]:],
                     skip_special_tokens=False).strip()
    # registered specials always decode as their strings — normalize to
    # numeric so data tokens 256-259 compare correctly
    for s, i in [("<pad>", 256), ("<bos>", 257), ("<sep>", 258),
                 ("<eos>", 259)]:
        gen = gen.replace(s, str(i))
    want = trg.strip() + " 259"
    ok = gen == want
    correct += ok
    if i < 3 or not ok:
        print(("MATCH" if ok else "MISS "), "gen:", gen)
        if not ok:
            print("      want:", want)
print(f"\n{correct}/{N_SAMPLES} exact matches via HF generate()")
assert correct >= int(0.9 * N_SAMPLES), "HF serving does not reproduce the model"
print("SERVING VERIFIED")
