#!/usr/bin/env bash
#
# run_task.sh --task <instance_id> --mode <docker|native> [--gold-first]
#
# Run one SWE-PolyBench task through the Pi coding agent and capture everything
# needed to reconstruct the result.
#
# The core loop is identical in both modes:
#   1. Reset repo to a clean base: git reset --hard && git clean -fdx, then
#      checkout the task's base_commit.
#   2. pi -p "<problem_statement>" --mode json  (full JSON log captured).
#   3. Extract the git diff as the candidate patch.
#   4. Save patch + JSON log + a run record under results/<model>/<task_id>/.
#
# --gold-first  : before spending an agent run, check the task is scoreable AT
#                 ALL by grading the dataset's own gold patch. If gold cannot be
#                 scored here, neither can any model's patch, and the run would
#                 produce an ERROR rather than a result. Exits 3 without running
#                 the agent. Costs one image pull, which the candidate
#                 verification then reuses (the image is kept).
#
# --mode native : run the loop directly on the current filesystem (RunPod, no
#                 Docker; isolation comes from the git reset/clean above).
# --mode docker : build the x86 test image with the task repo baked in, then
#                 run THIS SAME SCRIPT in --mode native inside the container.
#                 (Used on Mac for Mode A. RunPod can't do Docker-in-Docker.)
#
# Model/provider come from env or flags; Pi's provider config is separate (see
# run/pi_config/) so the harness can later vary while the model is held fixed.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

TASK=""
MODE="native"
MODEL="${MODEL:-anthropic/claude-sonnet-5}"
PROVIDER="${PROVIDER:-openrouter}"
WORKDIR="${WORKDIR:-}"                 # repo checkout to operate on (native mode)
IMAGE="${IMAGE:-agent-serving-bench:test}"
PI_EXTENSION="${PI_EXTENSION:-}"       # optional harness/ context strategy (future)
GOLD_FIRST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --gold-first) GOLD_FIRST=1; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --extension) PI_EXTENSION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" ]] || { echo "usage: run_task.sh --task <id> --mode <docker|native>" >&2; exit 2; }

# Task records live in one of two datasets. SWE-PolyBench is checked first so
# existing behaviour is untouched; SWE-bench Lite is a fallback, which keeps
# this additive -- no PolyBench task resolves differently than before.
TASK_JSON="$REPO_ROOT/tasks/data/${TASK}.json"
if [[ ! -f "$TASK_JSON" && -f "$REPO_ROOT/tasks/swebench_lite/data/${TASK}.json" ]]; then
  TASK_JSON="$REPO_ROOT/tasks/swebench_lite/data/${TASK}.json"
fi
[[ -f "$TASK_JSON" ]] || { echo "no task file: $TASK_JSON (run tasks/fetch_tasks.py)" >&2; exit 1; }

task_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$TASK_JSON" "$1"; }
REPO="$(task_field repo)"
BASE_COMMIT="$(task_field base_commit)"

# results/<model-slug>/<task_id>/
MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"
OUT="$REPO_ROOT/results/${MODEL_SLUG}/${TASK}"

# An episode is (task, model, attempt). A retry must not overwrite the previous
# attempt -- comparing them is the whole reason to retry. Any existing record is
# moved aside as <task>__aN before this run writes, so every attempt keeps its
# own directory with its own pi_log, patch, metrics and verdict.
#
# Skipped inside the container: docker mode re-enters this script as native, and
# archiving there would file away the directory the outer invocation just made.
if [[ -z "${RUN_DEPLOYMENT:-}" && -f "$OUT/run_record.json" ]]; then
  n=1
  while [[ -e "${OUT}__a${n}" ]]; do n=$((n + 1)); done
  mv "$OUT" "${OUT}__a${n}"
  echo "[run] previous attempt archived -> $(basename "${OUT}__a${n}")"
fi

# =========================================================================
# GOLD-FIRST PREFLIGHT: is this task scoreable here at all?
# =========================================================================
# Grading the gold patch is independent of any model, so a failure means the
# task cannot yield a result on this machine -- wrong CPU for its native deps,
# a test runner we cannot parse, a missing image. Finding that out first turns
# a wasted agent run plus a slow image pull into just the pull.
if [[ "$GOLD_FIRST" -eq 1 ]]; then
  GOLD_JSON="$REPO_ROOT/results/_gold/${TASK}/verify.json"
  gold_ok() {
    python3 -c 'import json,sys
try:
    sys.exit(0 if json.load(open(sys.argv[1])).get("resolved") is True else 1)
except Exception:
    sys.exit(1)' "$GOLD_JSON" 2>/dev/null
  }

  if gold_ok; then
    echo "[run] gold-first: cached PASS for $TASK (results/_gold) -- proceeding"
  elif ! docker info >/dev/null 2>&1; then
    # Mode B has no Docker (that is the whole reason it runs native), so the
    # preflight cannot run there. Warn and continue rather than block the run.
    echo "[run] gold-first: no docker daemon -- cannot preflight; continuing anyway" >&2
  else
    echo "[run] gold-first: grading gold patch for $TASK before spending an agent run"
    # Keep the image: the candidate verification right after this needs the
    # same one, and these pulls are the slowest step in the loop.
    "$HERE/verify_task.sh" --task "$TASK" --gold --keep-image || true
    if gold_ok; then
      echo "[run] gold-first: PASS -- task is scoreable, proceeding"
    else
      echo "[run] gold-first: FAIL -- gold patch could not be scored on this machine." >&2
      echo "[run] Skipping the agent run; an ERROR here would say nothing about the model." >&2
      echo "[run] See $GOLD_JSON  (Mode B / native x86 may score it fine)" >&2
      exit 3
    fi
  fi
fi

# =========================================================================
# DOCKER MODE (Mac / Mode A): build image with the repo baked in, then
# re-enter this script as native inside the container.
# =========================================================================
if [[ "$MODE" == "docker" ]]; then
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY for docker/OpenRouter mode}"
  CLONE_URL="https://github.com/${REPO}.git"
  echo "[run] docker mode: building image for repo=$REPO"
  docker build --platform linux/amd64 \
    -f "$REPO_ROOT/Dockerfile.test" \
    --build-arg TASK_REPO_URL="$CLONE_URL" \
    -t "$IMAGE" "$REPO_ROOT"

  echo "[run] docker mode: running task $TASK inside container"
  mkdir -p "$OUT"
  docker run --rm --platform linux/amd64 \
    -e OPENROUTER_API_KEY \
    -e MODEL="$MODEL" \
    -e PROVIDER="$PROVIDER" \
    -e PI_EXTENSION="$PI_EXTENSION" \
    -e RUN_DEPLOYMENT=docker \
    -v "$REPO_ROOT/results":/agent/results \
    -v "$REPO_ROOT/tasks":/agent/tasks \
    -v "$REPO_ROOT/run":/agent/run:ro \
    -v "$REPO_ROOT/run/pi_config/models.openrouter.json":/root/.pi/agent/models.json:ro \
    "$IMAGE" \
    /agent/run/run_task.sh --task "$TASK" --mode native --workdir /repo
  echo "[run] docker mode: done -> $OUT"
  exit 0
fi

# =========================================================================
# NATIVE MODE (RunPod, or inside the container): the actual loop.
# =========================================================================
: "${WORKDIR:?native mode needs --workdir <repo checkout path>}"
[[ -d "$WORKDIR/.git" ]] || { echo "no git repo at $WORKDIR" >&2; exit 1; }

echo "[run] task=$TASK model=$MODEL provider=$PROVIDER mode=native"
echo "[run] repo=$REPO base_commit=$BASE_COMMIT workdir=$WORKDIR"

# 1. Reset to a clean base. This is the isolation boundary in native mode.
cd "$WORKDIR"
git config --global --add safe.directory "$WORKDIR" 2>/dev/null || true
git reset --hard
git clean -fdx
git checkout --quiet "$BASE_COMMIT"
echo "[run] repo reset to $BASE_COMMIT"

mkdir -p "$OUT"

# 2. Run Pi. Problem statement is passed straight through; --no-session keeps
#    runs independent. --mode json streams every event as JSON lines.
PROBLEM="$(python3 -c 'import json,sys; sys.stdout.write(json.load(open(sys.argv[1]))["problem_statement"])' "$TASK_JSON")"

PI_ARGS=( -p "$PROBLEM" --mode json --provider "$PROVIDER" --model "$MODEL" --no-session )
[[ -n "$PI_EXTENSION" ]] && PI_ARGS+=( --extension "$PI_EXTENSION" )

echo "[run] invoking: pi -p <problem> --mode json --provider $PROVIDER --model $MODEL"
START="$(date +%s)"
set +e
# Pipe Pi's event stream through the timestamper: raw log -> pi_log.jsonl,
# capture-time timestamps -> events.timed.jsonl (needed for TTFT).
pi "${PI_ARGS[@]}" 2>"$OUT/pi_stderr.log" \
  | python3 -u "$HERE/_stamp_stream.py" "$OUT/events.timed.jsonl" >"$OUT/pi_log.jsonl"
PI_RC=${PIPESTATUS[0]}
set -e
END="$(date +%s)"
WALL=$(( END - START ))
echo "[run] pi exited rc=$PI_RC wall=${WALL}s"

# 3. Extract candidate patch (everything changed vs base_commit, incl new files).
# Diff against base_commit (not HEAD) so we still capture the fix even if the
# agent committed it (HEAD would then hide it from `git diff --cached`).
git add -A
git diff --cached "$BASE_COMMIT" > "$OUT/candidate.patch" || true
PATCH_BYTES="$(wc -c < "$OUT/candidate.patch" | tr -d ' ')"
echo "[run] candidate patch: ${PATCH_BYTES} bytes -> $OUT/candidate.patch"

# 4. Run record: reconstruct exactly what produced this result.
TS="$(date -u +%Y%m%dT%H%M%SZ)"
HARNESS_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
# Deployment: docker mode sets RUN_DEPLOYMENT=docker in the container; a direct
# native run (RunPod, no Docker) leaves it unset -> "runpod".
DEPLOY="${RUN_DEPLOYMENT:-runpod}"
python3 - "$OUT/run_record.json" "$TASK" "$REPO" "$BASE_COMMIT" "$MODEL" \
    "$PROVIDER" "$MODE" "$PI_RC" "$WALL" "$PATCH_BYTES" "$TS" "$HARNESS_SHA" \
    "$PI_EXTENSION" "$DEPLOY" <<'PY'
import json, sys
(path, task, repo, base, model, provider, mode, rc, wall, patch_bytes, ts,
 harness_sha, extension, deployment) = sys.argv[1:15]
record = {
    "timestamp_utc": ts,
    "task_id": task,
    "repo": repo,
    "base_commit": base,
    "mode": mode,
    "deployment": deployment,             # docker (Mode A) | runpod (Mode B)
    "model": model,
    "provider": provider,
    "pi_extension": extension or None,   # harness context strategy, if any
    "pi_exit_code": int(rc),
    "wall_seconds": int(wall),
    "patch_bytes": int(patch_bytes),
    "patch_produced": int(patch_bytes) > 0,
    "harness_git_sha": harness_sha,
    "pi_command": [
        "pi", "-p", "<problem_statement>", "--mode", "json",
        "--provider", provider, "--model", model, "--no-session",
    ] + (["--extension", extension] if extension else []),
}
with open(path, "w") as f:
    json.dump(record, f, indent=2)
print(f"[run] run record -> {path}")
PY

# Metrics, including the model-time / tool-time split, are written here so every
# run carries them. Reconstructing the split afterwards from the raw stream was
# how a 74-minute post-agent pipe hang got mistaken for 74 minutes of work.
python3 "$HERE/extract_metrics.py" "$OUT" >/dev/null 2>&1 \
  && echo "[run] metrics -> $OUT/metrics.json" \
  || echo "[run] metrics extraction skipped (see $OUT/pi_log.jsonl)"

echo "[run] results in $OUT"
[[ "$PI_RC" -eq 0 ]] || echo "[run] NOTE: pi returned non-zero; see $OUT/pi_stderr.log" >&2
exit 0
