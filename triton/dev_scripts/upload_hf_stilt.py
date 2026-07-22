"""Upload a verified Stilt staging dir to the HF Hub (private-first).

Policy (thesis/STILT-ROADMAP.md): minimal card, no token bounds,
handler.py included, private on first push. Login-node script.

Usage: python upload_hf_stilt.py <staging_dir> <repo_id> [--public]
"""
import sys

from huggingface_hub import HfApi

REQUIRED = ["config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "README.md", "handler.py",
            "configuration_stieltjes_gpt2.py",
            "modeling_stieltjes_gpt2.py"]


def main():
    staging, repo_id = sys.argv[1], sys.argv[2]
    public = "--public" in sys.argv[3:]
    import os
    missing = [f for f in REQUIRED if not os.path.exists(
        os.path.join(staging, f))]
    if missing:
        sys.exit(f"staging incomplete, missing: {missing}")
    api = HfApi()
    api.create_repo(repo_id, private=not public, exist_ok=True)
    api.upload_folder(folder_path=staging, repo_id=repo_id,
                      ignore_patterns=["__pycache__/*", "*.pyc"])
    files = api.list_repo_files(repo_id)
    print(f"uploaded {len(files)} files -> "
          f"https://huggingface.co/{repo_id}")
    for f in sorted(files):
        print(" ", f)


if __name__ == "__main__":
    main()
