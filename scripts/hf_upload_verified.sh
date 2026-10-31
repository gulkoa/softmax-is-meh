#!/bin/bash
# Upload every staging dir in LIST whose convert.log says VERIFIED and that
# has not been uploaded yet (marker <staging>/.uploaded holding the repo
# url), privately, with triton/dev_scripts/upload_hf_stilt.py. Login node.
# Written 2026-10-30 for the research-checkpoint upload.
#   bash scripts/hf_upload_verified.sh LIST
set -u
LIST=$1
R=/users/PAS2402/alexg/softmax/softmax-is-meh
PY=$R/triton/.venv/bin/python
n=0; skipped=0; failed=0
while read -r CKPT STAGING TRAINER REPO; do
    [ -n "$CKPT" ] || continue
    if [ -f "$STAGING/.uploaded" ]; then skipped=$((skipped + 1)); continue; fi
    if ! grep -q "^VERIFIED" "$STAGING/convert.log" 2>/dev/null; then
        echo "not verified: $REPO"; skipped=$((skipped + 1)); continue
    fi
    if out=$($PY "$R/triton/dev_scripts/upload_hf_stilt.py" "$STAGING" "$REPO" 2>&1 | grep -E "^uploaded"); then
        echo "$out"; echo "https://huggingface.co/$REPO" > "$STAGING/.uploaded"; n=$((n + 1))
    else
        echo "upload FAILED: $REPO"; failed=$((failed + 1))
    fi
done < "$LIST"
echo "uploaded $n, skipped $skipped, failed $failed"
