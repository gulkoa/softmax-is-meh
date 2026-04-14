#!/bin/bash
# Quick status check: newest messages, squeue, tail of latest ascend retrain log
set -u
MSG_DIR="/users/PAS2402/alexg/softmax/messages"
RES_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh/results"

echo "=== Newest messages ==="
ls -1t "$MSG_DIR" | head -5

echo
echo "=== squeue ==="
squeue -u "$USER" 2>/dev/null

echo
echo "=== Latest ascend retrain logs ==="
ls -1t "$RES_DIR"/nanogpt-*ascend-retrain*.out 2>/dev/null | head -3

echo
echo "=== Tail of most recent ==="
latest=$(ls -1t "$RES_DIR"/nanogpt-*ascend-retrain*.out 2>/dev/null | head -1)
if [ -n "$latest" ]; then
  echo "File: $latest"
  tail -20 "$latest"
fi
