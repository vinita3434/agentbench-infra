#!/usr/bin/env bash
#
# collect_and_verify.sh --model <registry-key> [--pod user@host] [--port N]
#                       [--remote-root PATH] [--no-pull] [--keep-image]
#
# Bring pod results home and score them. Verification needs Docker, which the
# pod cannot provide, so patches are produced there and graded here.
#
# Idempotent by design: a task that already has a verify.json is skipped, so
# re-running after adding tasks (or after a failed pull) only does the new work.
# Safe to run mid-sweep.
#
# Only the patch has to travel. candidate.patch is a diff against the task's
# base_commit, and the eval image already contains that repo at that commit --
# so where the patch was produced is irrelevant to scoring it.
set -uo pipefail   # not -e: one bad task must not abandon the rest

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

MODEL=""
POD=""
PORT=""
REMOTE_ROOT="~/agentbench-infra"
PULL=1
KEEP_IMAGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --no-pull) PULL=0; shift ;;
    --keep-image) KEEP_IMAGE=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "usage: collect_and_verify.sh --model <registry-key> [--pod user@host]" >&2; exit 2; }

# Results dirs are slugged the same way run_task.sh and run_pod.sh slug them:
# `tr '/:' '__'` maps each character to a SINGLE underscore. A pod label
# (qwen3-coder-30b) is unchanged; a frontier model string is not.
MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"
LOCAL_DIR="$ROOT/results/$MODEL_SLUG"

# --- 1. pull ---------------------------------------------------------------
if [[ "$PULL" -eq 1 && -n "$POD" ]]; then
  SSH_OPT=()
  [[ -n "$PORT" ]] && SSH_OPT=(-e "ssh -p $PORT")
  echo "[collect] pulling results/$MODEL_SLUG from $POD"
  mkdir -p "$LOCAL_DIR"
  rsync -az "${SSH_OPT[@]}" "$POD:$REMOTE_ROOT/results/$MODEL_SLUG/" "$LOCAL_DIR/" \
    || echo "[collect] WARNING: results rsync failed; verifying whatever is already local" >&2
  # Serving metrics are the actual deliverable of this phase -- pull them even
  # if the run results failed to transfer.
  echo "[collect] pulling results/serving"
  mkdir -p "$ROOT/results/serving"
  rsync -az "${SSH_OPT[@]}" "$POD:$REMOTE_ROOT/results/serving/" "$ROOT/results/serving/" \
    || echo "[collect] WARNING: serving-metrics rsync failed" >&2
elif [[ "$PULL" -eq 1 ]]; then
  echo "[collect] no --pod given; verifying local results only"
fi

[[ -d "$LOCAL_DIR" ]] || { echo "[collect] nothing at $LOCAL_DIR" >&2; exit 1; }

# --- 2. verify what needs it ----------------------------------------------
docker info >/dev/null 2>&1 || { echo "[collect] docker daemon is not running" >&2; exit 1; }

VERIFY_ARGS=(--model "$MODEL")
[[ "$KEEP_IMAGE" -eq 1 ]] && VERIFY_ARGS+=(--keep-image)

todo=(); skipped=0
for dir in "$LOCAL_DIR"/*/; do
  [[ -d "$dir" ]] || continue
  task="$(basename "$dir")"
  if [[ ! -s "$dir/candidate.patch" ]]; then
    # No patch means the agent produced nothing. Guaranteed not-resolved, and
    # verify_task.sh refuses it -- so don't spend a multi-GB pull to confirm.
    echo "[collect] skip $task: no candidate patch"
    skipped=$((skipped + 1))
    continue
  fi
  if [[ -f "$dir/verify.json" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  todo+=("$task")
done

echo "[collect] ${#todo[@]} task(s) to verify, $skipped already done or skipped"

pass=0; fail=0
for task in "${todo[@]:-}"; do
  [[ -n "$task" ]] || continue
  echo
  echo "=== [collect] verify $task ==========================="
  if "$HERE/verify_task.sh" --task "$task" "${VERIFY_ARGS[@]}"; then
    pass=$((pass + 1))
  else
    # A failed verification is a data point, not a reason to stop the batch.
    fail=$((fail + 1))
    echo "[collect] verify_task.sh returned non-zero for $task (continuing)" >&2
  fi
done

# --- 3. roll up ------------------------------------------------------------
echo
"$HERE/status.sh" --model "$MODEL" --save "$ROOT/results/${MODEL_SLUG}_tasks.txt"
echo
echo "[collect] verified $pass, errored $fail, skipped $skipped"
echo "[collect] serving metrics: $ROOT/results/serving/$MODEL_SLUG/"
