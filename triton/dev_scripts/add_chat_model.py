"""Add a published -it model to stiltchat: create a dedicated Inference
Endpoint cloned from an existing chat endpoint's instance config, insert
it into the Space's MODELS registry, push the static Space via git.

Needs a WRITE token in HF_TOKEN (or the default hf login). PAID: a new
endpoint is a user spend decision — run only on explicit direction
(2026-08-16: "add it to chat" for stilt.1.1-355m-nope-it).

Usage: python add_chat_model.py GulkoA/stilt.1.1-355m-nope-it \
           --label "stilt.1.1-355m-nope-it (long context)" \
           [--clone-from stilt-1-355m-it] [--dry-run]
"""
import argparse, os, re, subprocess, sys, time
from huggingface_hub import HfApi, get_token

SPACE = "GulkoA/stiltchat"
STAGING = "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_space_staging"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--label", required=True)
    ap.add_argument("--clone-from", default="stilt-1-355m-it")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    ref = api.get_inference_endpoint(a.clone_from)
    cfg = ref.raw
    comp, prov = cfg["compute"], cfg["provider"]
    name = a.repo.split("/")[1].replace(".", "-")
    print(f"clone config from {a.clone_from}: {prov['vendor']}/{prov['region']} "
          f"{comp['accelerator']} {comp['instanceType']} {comp['instanceSize']} "
          f"scale_to_zero={comp.get('scaling', {}).get('scaleToZeroTimeout')}")
    if a.dry_run:
        print("dry-run: would create endpoint", name); return
    ep = api.create_inference_endpoint(
        name, repository=a.repo, framework=cfg["model"]["framework"],
        task=cfg["model"]["task"], accelerator=comp["accelerator"],
        instance_size=comp["instanceSize"], instance_type=comp["instanceType"],
        region=prov["region"], vendor=prov["vendor"],
        type=cfg.get("type", "protected"),
        min_replica=comp["scaling"]["minReplica"],
        max_replica=comp["scaling"]["maxReplica"],
        scale_to_zero_timeout=comp["scaling"].get("scaleToZeroTimeout"),
        custom_image=cfg["model"].get("image", {}).get("custom"))
    print("created endpoint", ep.name, "-> waiting for URL")
    for _ in range(60):
        ep = api.get_inference_endpoint(name)
        if ep.url: break
        time.sleep(10)
    assert ep.url, "endpoint has no URL yet — rerun the Space patch later"
    print("url:", ep.url)
    # patch the Space registry (insert after the first entry so it is
    # visible but not the default)
    p = os.path.join(STAGING, "index.html")
    s = open(p).read()
    entry = (f'  {{ id: "{a.repo}",\n    label: "{a.label}",\n'
             f'    endpoint:\n      "{ep.url}" }},\n')
    m = re.search(r"const MODELS = \[\n(.*?\}\,\n)", s, re.S)
    assert m, "MODELS block not found"
    s = s[:m.end()] + entry + s[m.end():]
    open(p, "w").write(s)
    tok = os.environ.get("HF_TOKEN") or get_token()
    url = f"https://GulkoA:{tok}@huggingface.co/spaces/{SPACE}"
    # staging sits inside the code monorepo: push from a throwaway clone
    import shutil, tempfile
    tmp = tempfile.mkdtemp(prefix="stiltchat_", dir=os.environ.get(
        "SCRATCHPAD", "/tmp/claude-41244/-users-PAS2402-alexg-softmax/"
        "89d1d46a-ca91-4553-b27b-596dc1e35dfb/scratchpad"))
    subprocess.run(["git", "clone", "-q", url, tmp], check=True)
    shutil.copy(p, os.path.join(tmp, "index.html"))
    subprocess.run(["git", "-C", tmp, "add", "index.html"], check=True)
    subprocess.run(["git", "-C", tmp, "-c", "user.name=alexg",
                    "-c", "user.email=alexgulko99@gmail.com", "commit", "-qm",
                    f"chat: add {a.repo}"], check=True)
    r = subprocess.run(["git", "-C", tmp, "push", "-q", "origin", "HEAD"],
                       capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("space push:", "ok" if r.returncode == 0 else
          r.stderr.replace(tok, "<token>")[-300:])

if __name__ == "__main__":
    main()
