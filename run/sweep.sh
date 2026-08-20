#!/usr/bin/env bash
#
# sweep.sh --model <slug> --tier <easy|hard|all> [--dataset swebench|polybench]
#          [--min-free-gb N] [--dry-run]
#
# Run a tier of tasks end to end: agent, verify, garbage-collect, repeat.
#
# The gc step is the point. A previous sweep filled the disk to 98% and Docker
# Desktop then refused to start -- and once the daemon is down you cannot prune
# your way out. Cleaning after every task keeps usage flat instead of monotonic,
# and the disk floor stops the sweep while the results so far are still intact.
#
# Idempotent: a task that already has a verify.json is skipped, so re-running
# after an interruption only does the remaining work.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

MODEL=""; TIER="all"; DATASET="swebench"; MIN_FREE=12; DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "usage: sweep.sh --model <slug> --tier <easy|hard|all>" >&2; exit 2; }

if [[ "$DATASET" == "swebench" ]]; then
  VERIFY="$HERE/verify_swebench.sh"
  TASKS=$(python3 -c "
import json
d = json.load(open('$ROOT/tasks/swebench_lite/tiers.json'))
tier = '$TIER'
print(' '.join(d['easy'] + d['hard'] if tier == 'all' else d[tier]))")
else
  VERIFY="$HERE/verify_task.sh"
  TASKS=$(grep '^[a-z]' "$ROOT/results/resolved_tasks.txt" | tr '\n' ' ')
fi

MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"
total=$(wc -w <<<"$TASKS" | tr -d ' ')
echo "[sweep] $MODEL · $DATASET · tier=$TIER · $total task(s)"
[[ "$DRY" -eq 1 ]] && { printf '  %s\n' $TASKS; exit 0; }

i=0; ran=0; skipped=0
for task in $TASKS; do
  i=$((i + 1))
  if [[ -f "$ROOT/results/$MODEL_SLUG/$task/verify.json" ]]; then
    skipped=$((skipped + 1)); continue
  fi

  # Check headroom BEFORE starting, so a task never dies half-way through for
  # want of disk -- that loses the trajectory as well as the disk.
  if ! "$HERE/docker_gc.sh" --min-free-gb "$MIN_FREE" >/dev/null 2>&1; then
    "$HERE/docker_gc.sh" --aggressive --min-free-gb "$MIN_FREE" || {
      echo "[sweep] stopping: below the ${MIN_FREE}GB floor even after cleanup" >&2
      break
    }
  fi

  echo; echo "=== [sweep] ($i/$total) $task ==============================="
  "$HERE/run_task.sh" --task "$task" --mode docker --model "$MODEL" || {
    echo "[sweep] agent run failed for $task, continuing" >&2
    "$HERE/docker_gc.sh" >/dev/null 2>&1; continue; }
  "$VERIFY" --task "$task" --model "$MODEL" || \
    echo "[sweep] verify returned non-zero for $task, continuing" >&2
  ran=$((ran + 1))
  # Aggressive by default: drop the eval image as soon as its task is scored.
  # Keeping them accumulated 20GB of images for two tasks that never ran, and a
  # full disk is what stops Docker starting at all. A re-pull only costs time,
  # and only if the same task is verified twice.
  "$HERE/docker_gc.sh" --aggressive
done

echo; echo "[sweep] ran $ran, skipped $skipped (already verified)"
python3 "$HERE/master_table.py" 2>/dev/null | tail -3
