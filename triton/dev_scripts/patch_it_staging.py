"""Post-convert patch for -it staging dirs: inject the Stilt chat
template into tokenizer_config.json (the converter regenerates vanilla
GPT-2 tokenizer files, which lack it), install the CHAT handler (base
stagings carry the completion-only handler — wrong handler = float
'inputs' tensor crash on the endpoint, hit 2026-07-22), and stamp the
model name into it. No-token-bounds values are already handled by the
converter. Login-node script.

Usage: python patch_it_staging.py <staging_dir> <model_name>
  e.g. python patch_it_staging.py results/hf_355m_it_staging stilt.1-355m-it
"""
import json
import os
import re
import sys

CHAT_HANDLER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "results", "hf_it_staging",
                            "handler.py")

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] in ['user', 'system'] %}"
    "<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

# v2 (-it2 models, trained WITH a system channel): native <|system|>
CHAT_TEMPLATE_V2 = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|system|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}"
    "<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

FOLD_BRANCH = ('                # no trained system channel; fold into '
               'the first user turn\n'
               '                prompt += U + "(instructions) " + '
               'content + "\\n"')
NATIVE_BRANCH = ('                prompt += "<|system|>\\n" + content '
                 '+ "\\n"')


def main():
    staging, model_name = sys.argv[1], sys.argv[2]
    system_native = "--system-native" in sys.argv[3:]
    p = f"{staging}/tokenizer_config.json"
    tc = json.load(open(p))
    tc["chat_template"] = (CHAT_TEMPLATE_V2 if system_native
                           else CHAT_TEMPLATE)
    json.dump(tc, open(p, "w"), indent=2)
    assert json.load(open(p))["model_max_length"] > 10**9, "bounds crept back"
    print(f"chat template injected (v2={system_native}) -> {p}")

    src = open(CHAT_HANDLER).read()
    assert '"messages"' in src, "canonical chat handler lost its chat path?"
    patched = re.sub(r'"model": data\.get\("model", "[^"]*"\)',
                     f'"model": data.get("model", "{model_name}")', src)
    if system_native:
        assert FOLD_BRANCH in patched, "handler fold branch drifted"
        patched = patched.replace(FOLD_BRANCH, NATIVE_BRANCH)
    with open(f"{staging}/handler.py", "w") as f:
        f.write(patched)
    print(f"chat handler installed ({model_name}, "
          f"system_native={system_native}) -> {staging}/handler.py")


if __name__ == "__main__":
    main()
