"""Post-convert patch for -it staging dirs: inject the Stilt chat
template into tokenizer_config.json (the converter regenerates vanilla
GPT-2 tokenizer files, which lack it). No-token-bounds values are
already handled by the converter. Login-node script.

Usage: python patch_it_staging.py <staging_dir>
"""
import json
import sys

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] in ['user', 'system'] %}"
    "<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def main():
    staging = sys.argv[1]
    p = f"{staging}/tokenizer_config.json"
    tc = json.load(open(p))
    tc["chat_template"] = CHAT_TEMPLATE
    json.dump(tc, open(p, "w"), indent=2)
    assert json.load(open(p))["model_max_length"] > 10**9, "bounds crept back"
    print(f"chat template injected -> {p}")


if __name__ == "__main__":
    main()
