"""Local LLM-judge reward for chat GRPO (plan: thesis/findings/
2026-07-22-plan-judge-rl-chat.md). Qwen2.5-7B-Instruct on the same
GPU as the policy: greedy decode (deterministic), strict-JSON scores
with one parse-retry, plus a judge-independent string-match scorer for
the gold-anchored TriviaQA canary channel.

Smoke test (GPU): python judge_reward.py --smoke
"""
import json
import os
import re
import string
import sys

import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import FW_DIR  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

RUBRIC = """You grade answers from a small research language model. \
Score the ANSWER to the QUESTION on a 1-10 scale:
- 9-10: correct, clear, directly answers the question
- 6-8: mostly correct, minor errors or vagueness
- 3-5: partially relevant but substantially wrong or confused
- 1-2: wrong, off-topic, gibberish, or empty

Rules: judge substance, NOT length — do not reward verbosity, \
hedging, boilerplate openers ("Great question!"), or self-reference. \
A short correct answer beats a long padded one.

Reply with ONLY this JSON, nothing else:
{"score": <int 1-10>, "reason": "<one short sentence>"}"""


class LocalJudge:
    def __init__(self, model_name=JUDGE_MODEL, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # plain .to(device) — device_map needs accelerate, not in venv
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()

    @torch.no_grad()
    def _generate(self, chats, max_new=60):
        texts = [self.tok.apply_chat_template(c, tokenize=False,
                                              add_generation_prompt=True)
                 for c in chats]
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       padding_side="left").to(self.model.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.tok.eos_token_id)
        return [self.tok.decode(out[i, enc.input_ids.shape[1]:],
                                skip_special_tokens=True)
                for i in range(len(chats))]

    @staticmethod
    def _parse(text):
        m = re.search(r'\{[^{}]*"score"\s*:\s*(\d+)[^{}]*\}', text)
        if not m:
            return None
        s = int(m.group(1))
        return s if 1 <= s <= 10 else None

    def score(self, questions, answers, max_retry=1):
        """Returns (scores list of float|None, parse_fail_count)."""
        chats = [[{"role": "system", "content": RUBRIC},
                  {"role": "user",
                   "content": f"QUESTION: {q}\n\nANSWER: {a}"}]
                 for q, a in zip(questions, answers)]
        outs = self._generate(chats)
        scores = [self._parse(o) for o in outs]
        for _ in range(max_retry):
            bad = [i for i, s in enumerate(scores) if s is None]
            if not bad:
                break
            outs2 = self._generate([chats[i] for i in bad])
            for i, o in zip(bad, outs2):
                scores[i] = self._parse(o)
        fails = sum(1 for s in scores if s is None)
        return scores, fails


def _norm(s):
    s = s.lower().strip()
    return " ".join(
        "".join(ch for ch in s if ch not in string.punctuation).split())


def gold_match(answer, aliases):
    """Judge-independent gold scorer: 1.0 if any gold alias appears in
    the normalized answer, else 0.0. The anti-hacking canary channel."""
    na = _norm(answer)
    return 1.0 if any(_norm(g) and _norm(g) in na for g in aliases) else 0.0


def smoke():
    judge = LocalJudge()
    qs = ["What is the capital of France?"] * 4 + [
        "Explain photosynthesis in one sentence."]
    ans = [
        "The capital of France is Paris.",                       # good
        "I think it might be Paris, which is a city in Europe, "
        "and cities are important. Great question!",             # padded
        "The capital of France is Berlin.",                      # wrong
        "banana banana banana banana",                           # gibberish
        "Photosynthesis is the process by which plants convert "
        "sunlight, water and CO2 into glucose and oxygen.",      # good
    ]
    scores, fails = judge.score(qs, ans)
    print("scores:", scores, "| parse fails:", fails)
    assert fails == 0, "judge JSON parse failed on smoke set"
    assert scores[0] >= 8, f"good answer scored {scores[0]}"
    assert scores[2] <= 4, f"wrong answer scored {scores[2]}"
    assert scores[3] <= 3, f"gibberish scored {scores[3]}"
    assert scores[0] > scores[1], "verbosity not penalized vs clean"
    assert scores[4] >= 8, f"good science answer scored {scores[4]}"
    print("gold_match sanity:",
          gold_match("It is Paris, of course.", ["Paris"]) == 1.0,
          gold_match("Berlin is the answer", ["Paris"]) == 0.0)
    print("SMOKE PASSED")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
