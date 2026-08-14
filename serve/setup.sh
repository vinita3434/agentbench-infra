#!/usr/bin/env bash
#
# setup.sh --model <key>
#
# Prepare a GPU pod to serve <key>. Run on a fresh pod, or when switching to a
# model that pins a different SGLang version.
#
#   1. Read the pinned sglang_version from serve/models.yaml.
#   2. Check what SGLang (if any) is currently installed.
#   3. If they mismatch, STOP and ask before switching -- never silently
#      reinstall, because a version swap invalidates prior run records.
#   4. Install the EXACT pinned version (never "latest").
#   5. Ensure the HuggingFace CLI is present and auth is handled.
#
# Safe to re-run: if the right version is already installed it does nothing.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="$HERE/_registry.py"

MODEL=""
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "usage: setup.sh --model <key> [--yes]" >&2; exit 2; }

# --- ensure PyYAML so _registry.py works ---------------------------------
python3 -c 'import yaml' 2>/dev/null || {
  echo "[setup] installing PyYAML (needed to read models.yaml)"
  pip install --quiet pyyaml
}

WANT_VERSION="$(python3 "$REGISTRY" "$MODEL" sglang_version)"
[[ -n "$WANT_VERSION" ]] || { echo "[setup] no sglang_version for $MODEL" >&2; exit 1; }
echo "[setup] model=$MODEL wants SGLang==$WANT_VERSION"

# --- what's installed now? -----------------------------------------------
CUR_VERSION="$(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || true)"

if [[ "$CUR_VERSION" == "$WANT_VERSION" ]]; then
  echo "[setup] SGLang $CUR_VERSION already installed -- nothing to do."
else
  if [[ -n "$CUR_VERSION" ]]; then
    echo ""
    echo "  !! SGLang VERSION MISMATCH"
    echo "     installed: $CUR_VERSION"
    echo "     required : $WANT_VERSION  (for model '$MODEL')"
    echo ""
    echo "     Switching will reinstall SGLang for the WHOLE pod and can"
    echo "     invalidate results produced under $CUR_VERSION."
    if [[ "$ASSUME_YES" -ne 1 ]]; then
      read -r -p "     Reinstall SGLang $CUR_VERSION -> $WANT_VERSION? [y/N] " reply
      case "$reply" in
        y|Y|yes|YES) : ;;
        *) echo "[setup] aborted -- no changes made."; exit 1 ;;
      esac
    fi
  else
    echo "[setup] no SGLang detected -- installing fresh."
  fi
  echo "[setup] pip install 'sglang[all]==$WANT_VERSION'"
  pip install "sglang[all]==${WANT_VERSION}"
fi

# --- HuggingFace CLI + auth ----------------------------------------------
command -v huggingface-cli >/dev/null 2>&1 || {
  echo "[setup] installing huggingface_hub CLI"
  pip install --quiet "huggingface_hub[cli]"
}

if huggingface-cli whoami >/dev/null 2>&1; then
  echo "[setup] HuggingFace: logged in as $(huggingface-cli whoami)"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] HuggingFace: logging in with \$HF_TOKEN"
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
  echo "[setup] HuggingFace: now $(huggingface-cli whoami)"
else
  echo "[setup] HuggingFace: NOT logged in and \$HF_TOKEN unset."
  echo "        Gated repos will 401. Set HF_TOKEN or run: huggingface-cli login"
fi

echo "[setup] done. Next: serve/download.sh --model $MODEL"
