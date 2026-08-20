#!/usr/bin/env bash
#
# sweep_pod.sh --model <registry-key> [--tasks FILE] [--repos DIR]
#              [--timeout SECONDS] [--dry-run]
#
# Run a list of tasks on the pod, one after another. Mode B's counterpart to
# sweep.sh -- but there is no verify step here: scoring needs Docker and RunPod
# cannot provide it. This produces candidate patches; they get scored later on a
# Docker host via run/collect_and_verify.sh.
#
# The per-task job this does that run_pod.sh cannot: pick the right checkout.
# run_pod.sh takes a single --workdir, and a 45-task sweep spans 14 repos, so
# the repo is read from each task record and mapped to /workspace/repos/<name>.
#
# Idempotent: a task with a candidate.patch already present is skipped, so an
# interrupted sweep resumes without redoing work.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

MODEL=""; TASKS=""; REPOS="/workspace/repos"; TIMEOUT=1800; DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --repos) REPOS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "usage: sweep_pod.sh --model <registry-key> [--tasks FILE]" >&2; exit 2; }

# Default: everything proven scoreable in both datasets.
if [[ -z "$TASKS" ]]; then
  TASKS="$(mktemp)"
  grep '^[a-z]' "$ROOT/results/resolved_tasks.txt" 2>/dev/null > "$TASKS" || true
  cat "$ROOT/tasks/swebench_lite/all_valid.txt" 2>/dev/null >> "$TASKS" || true
fi
[[ -s "$TASKS" ]] || { echo "no tasks to run" >&2; exit 1; }

# task -> repo directory, resolved from whichever dataset holds the record.
PLAN="$(ROOT="$ROOT" TASKS="$TASKS" REPOS="$REPOS" python3 - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["ROOT"]); repos = os.environ["REPOS"]
for line in open(os.environ["TASKS"]):
    tid = line.strip()
    if not tid or tid.startswith("#"):
        continue
    for sub in ("data", "swebench_lite/data"):
        p = root / "tasks" / sub / f"{tid}.json"
        if p.exists():
            repo = json.load(p.open())["repo"]
            print(f"{tid}\t{repos}/{repo.split('/')[-1]}")
            break
PY
)"

total=$(wc -l <<<"$PLAN" | tr -d ' ')
echo "[sweep-pod] $MODEL · $total task(s) · repos under $REPOS"

if [[ "$DRY" -eq 1 ]]; then
  echo "$PLAN" | awk -F'\t' '{printf "    %-34s %s\n", $1, $2}'
  # Flag missing checkouts before the GPU is burning money on them.
  missing=0
  while IFS=$'\t' read -r t w; do
    [[ -d "$w/.git" ]] || { echo "    MISSING CHECKOUT: $w (for $t)"; missing=$((missing+1)); }
  done <<<"$PLAN"
  [[ "$missing" -gt 0 ]] && echo "[sweep-pod] $missing checkout(s) absent -- run prepare_repos.sh first" >&2
  exit 0
fi

i=0; ran=0; skipped=0
while IFS=$'\t' read -r task workdir; do
  i=$((i + 1))
  [[ -n "$task" ]] || continue
  if [[ -s "$ROOT/results/$MODEL/$task/candidate.patch" ]]; then
    skipped=$((skipped + 1)); continue
  fi
  if [[ ! -d "$workdir/.git" ]]; then
    echo "[sweep-pod] ($i/$total) $task: no checkout at $workdir -- skipping" >&2
    continue
  fi
  echo; echo "=== [sweep-pod] ($i/$total) $task ==============================="
  "$HERE/run_pod.sh" --task "$task" --model "$MODEL" --workdir "$workdir" \
      --timeout "$TIMEOUT" || echo "[sweep-pod] $task failed, continuing" >&2
  ran=$((ran + 1))
done <<<"$PLAN"

echo; echo "[sweep-pod] ran $ran, skipped $skipped (already have a patch)"
echo "[sweep-pod] patches are NOT scored here -- Docker is not available on the pod."
echo "[sweep-pod] from a Docker host:  run/collect_and_verify.sh --model $MODEL --pod <user@host>"
