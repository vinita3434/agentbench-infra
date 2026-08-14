#!/usr/bin/env bash
#
# download.sh --model <key>
#
# Fetch a model's weights from HuggingFace into the local cache so launch.sh
# starts instantly (no first-request download stall). Safe to re-run --
# huggingface-cli skips already-present files.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="$HERE/_registry.py"

MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "usage: download.sh --model <key>" >&2; exit 2; }

HF_REPO="$(python3 "$REGISTRY" "$MODEL" hf_repo)"
[[ -n "$HF_REPO" ]] || { echo "[download] no hf_repo for $MODEL" >&2; exit 1; }

command -v huggingface-cli >/dev/null 2>&1 || {
  echo "[download] huggingface-cli missing -- run serve/setup.sh --model $MODEL first" >&2
  exit 1
}

echo "[download] model=$MODEL repo=$HF_REPO"
echo "[download] cache: ${HF_HOME:-$HOME/.cache/huggingface}"
huggingface-cli download "$HF_REPO" --exclude "*.pth" "original/*"
echo "[download] done. Next: serve/launch.sh --model $MODEL"
