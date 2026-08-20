#!/usr/bin/env bash
#
# run_pod.sh --task <instance_id> --model <registry-key> --workdir <checkout>
#            [--endpoint URL] [--timeout SECONDS] [--extension PATH]
#
# Mode B: run the Pi coding agent on a RunPod CPU against a model SGLang is
# serving locally. Deliberately standalone -- it shares no code with
# run_task.sh, so nothing here can disturb the Mode A (Docker + OpenRouter)
# pipeline that produced the existing baseline.
#
# It writes the SAME output files as Mode A into results/<registry-key>/<task>/,
# because the results directory -- not the code -- is the interface every other
# tool reads. verify_task.sh, status.sh, peek.sh, report.py and task_artifact.py
# therefore work against pod output with no changes.
#
# Sibling: run/run_task.sh (Mode A). A fix to the run loop in one is worth
# considering for the other.
#
# Prereqs on the pod: serve/setup.sh, serve/download.sh, serve/launch.sh have
# run, and Pi is installed (npm i -g @earendil-works/pi-coding-agent).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

TASK=""
MODEL=""                                  # serve/models.yaml key -> results label
WORKDIR=""
ENDPOINT="${ENDPOINT:-http://localhost:30000/v1}"
TIMEOUT="${TIMEOUT:-1800}"                # wall-clock cap; Pi has no --max-turns
PI_EXTENSION="${PI_EXTENSION:-}"
SERVED_NAME="${SERVED_NAME:-local-model}" # what launch.sh serves as; see below
PROVIDER="selfhosted"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --extension) PI_EXTENSION="$2"; shift 2 ;;
    --served-name) SERVED_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TASK"    ]] || { echo "usage: run_pod.sh --task <id> --model <registry-key> --workdir <checkout>" >&2; exit 2; }
[[ -n "$MODEL"   ]] || { echo "--model <registry-key> is required (see: serve/_registry.py --list)" >&2; exit 2; }
[[ -n "$WORKDIR" ]] || { echo "--workdir <repo checkout> is required" >&2; exit 2; }

# Two datasets. PolyBench first so existing behaviour is unchanged; SWE-bench
# Lite is a fallback, matching run_task.sh.
TASK_JSON="$REPO_ROOT/tasks/data/${TASK}.json"
if [[ ! -f "$TASK_JSON" && -f "$REPO_ROOT/tasks/swebench_lite/data/${TASK}.json" ]]; then
  TASK_JSON="$REPO_ROOT/tasks/swebench_lite/data/${TASK}.json"
fi
[[ -f "$TASK_JSON" ]] || { echo "no task file: $TASK_JSON (run tasks/fetch_tasks.py)" >&2; exit 1; }
[[ -d "$WORKDIR/.git" ]] || { echo "no git repo at $WORKDIR" >&2; exit 1; }

# --- validate the label against the registry ------------------------------
# The label decides the results directory, so a typo would silently start a
# fresh results tree that looks like a legitimate new model.
HF_REPO="$(python3 "$REPO_ROOT/serve/_registry.py" "$MODEL" hf_repo)" || exit 1

# --- isolated Pi config ---------------------------------------------------
# Pi reads its provider config from PI_CODING_AGENT_DIR (default ~/.pi/agent).
# Pointing it at a repo-local directory is what keeps this pipeline from
# touching the OpenRouter setup -- no `cp ... ~/.pi/agent/models.json` anywhere.
export PI_CODING_AGENT_DIR="$REPO_ROOT/run/pi_config/pod_home"
mkdir -p "$PI_CODING_AGENT_DIR"
if [[ ! -f "$PI_CODING_AGENT_DIR/models.json" ]]; then
  cp "$REPO_ROOT/run/pi_config/models.selfhosted.json" "$PI_CODING_AGENT_DIR/models.json"
  echo "[pod] seeded $PI_CODING_AGENT_DIR/models.json from models.selfhosted.json"
fi

# --- endpoint preflight ---------------------------------------------------
# Without this, a server that is down or still loading weights produces an
# empty patch -- indistinguishable from a model that cannot code.
if ! curl -sf --max-time 10 "${ENDPOINT%/}/models" >/dev/null 2>&1; then
  echo "[pod] no OpenAI-compatible endpoint at ${ENDPOINT%/}/models" >&2
  echo "[pod] start it first:  serve/launch.sh --model $MODEL" >&2
  exit 4
fi
echo "[pod] endpoint OK: ${ENDPOINT%/}  (serving as '$SERVED_NAME')"

task_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$TASK_JSON" "$1"; }
REPO="$(task_field repo)"
BASE_COMMIT="$(task_field base_commit)"

# results/<registry-key>/<task_id>/ -- keyed by the LABEL, not by the served
# name. Every model is served as "local-model" so the Pi config never changes
# when weights change; without this the whole sweep would overwrite itself.
OUT="$REPO_ROOT/results/${MODEL}/${TASK}"

# Same rule as Mode A: a retry archives the previous attempt rather than
# overwriting it, so every (task, model, attempt) keeps its own record.
if [[ -f "$OUT/run_record.json" ]]; then
  n=1
  while [[ -e "${OUT}__a${n}" ]]; do n=$((n + 1)); done
  mv "$OUT" "${OUT}__a${n}"
  echo "[pod] previous attempt archived -> $(basename "${OUT}__a${n}")"
fi
mkdir -p "$OUT"

echo "[pod] task=$TASK label=$MODEL repo=$REPO base=$BASE_COMMIT"
echo "[pod] workdir=$WORKDIR timeout=${TIMEOUT}s"

# --- 1. reset the checkout: this is the isolation boundary ----------------
cd "$WORKDIR"
git config --local --add safe.directory "$WORKDIR" 2>/dev/null || true
git reset --hard >/dev/null
git clean -fdx >/dev/null
git checkout --quiet "$BASE_COMMIT"
echo "[pod] repo reset to $BASE_COMMIT"

# --- 2. run Pi ------------------------------------------------------------
PROBLEM="$(python3 -c 'import json,sys; sys.stdout.write(json.load(open(sys.argv[1]))["problem_statement"])' "$TASK_JSON")"

PI_ARGS=( -p "$PROBLEM" --mode json --provider "$PROVIDER" --model "$SERVED_NAME" --no-session )
[[ -n "$PI_EXTENSION" ]] && PI_ARGS+=( --extension "$PI_EXTENSION" )

command -v pi >/dev/null || { echo "[pod] 'pi' is not on PATH -- npm install -g @earendil-works/pi-coding-agent" >&2; exit 5; }

# `timeout` is the only cap available: Pi exposes no --max-turns, and a model
# that never converges has no natural stopping point. It is GNU coreutils, so
# it exists on the pod but not on a stock Mac -- fall back rather than exiting
# 127 into a 0-byte patch, which is indistinguishable from a failed run.
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
else
  echo "[pod] WARNING: no timeout/gtimeout on PATH -- running UNCAPPED" >&2
fi

echo "[pod] invoking: pi -p <problem> --mode json --provider $PROVIDER --model $SERVED_NAME"
START="$(date +%s)"
set +e
if [[ -n "$TIMEOUT_BIN" ]]; then
  "$TIMEOUT_BIN" "${TIMEOUT}s" pi "${PI_ARGS[@]}" 2>"$OUT/pi_stderr.log" \
    | python3 -u "$HERE/_stamp_stream.py" "$OUT/events.timed.jsonl" >"$OUT/pi_log.jsonl"
else
  pi "${PI_ARGS[@]}" 2>"$OUT/pi_stderr.log" \
    | python3 -u "$HERE/_stamp_stream.py" "$OUT/events.timed.jsonl" >"$OUT/pi_log.jsonl"
fi
PI_RC=${PIPESTATUS[0]}
set -e
# 127 means the binary vanished mid-flight; never let that pass as a model result.
[[ "$PI_RC" -eq 127 ]] && echo "[pod] ERROR: command not found (rc=127) -- see $OUT/pi_stderr.log" >&2
END="$(date +%s)"
WALL=$(( END - START ))
TIMED_OUT=false
if [[ -z "$TIMEOUT_BIN" ]]; then
  TIMEOUT=0          # 0 == no cap was applied, recorded honestly in run_record
fi
if [[ "$PI_RC" -eq 124 ]]; then
  TIMED_OUT=true
  echo "[pod] pi hit the ${TIMEOUT}s cap -- recorded as timed_out, not as an error"
fi
echo "[pod] pi exited rc=$PI_RC wall=${WALL}s"

# --- 3. candidate patch ---------------------------------------------------
# Diff against base_commit (not HEAD) so a fix the agent committed still shows.
git add -A
git diff --cached "$BASE_COMMIT" > "$OUT/candidate.patch" || true
PATCH_BYTES="$(wc -c < "$OUT/candidate.patch" | tr -d ' ')"
echo "[pod] candidate patch: ${PATCH_BYTES} bytes -> $OUT/candidate.patch"

# --- 4. run record --------------------------------------------------------
# Same schema as Mode A plus pod provenance, so report.py/status.sh read both.
TS="$(date -u +%Y%m%dT%H%M%SZ)"
HARNESS_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
# Newest launch record for this model, written by serve/launch.sh -- ties the
# result to the exact SGLang version and flags that served it.
SGLANG_RECORD="$(ls -1t "$REPO_ROOT/results/serve/${MODEL}"-*.json 2>/dev/null | head -1 || true)"

python3 - "$OUT/run_record.json" "$TASK" "$REPO" "$BASE_COMMIT" "$MODEL" \
    "$PROVIDER" "$PI_RC" "$WALL" "$PATCH_BYTES" "$TS" "$HARNESS_SHA" \
    "$PI_EXTENSION" "$SERVED_NAME" "$ENDPOINT" "$TIMED_OUT" "$TIMEOUT" \
    "$SGLANG_RECORD" "$HF_REPO" <<'PY'
import json, sys
(path, task, repo, base, label, provider, rc, wall, patch_bytes, ts,
 harness_sha, extension, served, endpoint, timed_out, timeout_s,
 sglang_record, hf_repo) = sys.argv[1:19]
record = {
    "timestamp_utc": ts,
    "task_id": task,
    "repo": repo,
    "base_commit": base,
    "mode": "native",
    "deployment": "runpod",
    # `model` is the label the results tree is keyed by; `served_name` is what
    # Pi actually asked the endpoint for. They differ on purpose.
    "model": label,
    "run_label": label,
    "hf_repo": hf_repo,
    "served_name": served,
    "endpoint": endpoint,
    "sglang_record": sglang_record or None,
    "provider": provider,
    "pi_extension": extension or None,
    "pi_exit_code": int(rc),
    "timed_out": timed_out == "true",
    "timeout_s": int(timeout_s),
    "wall_seconds": int(wall),
    "patch_bytes": int(patch_bytes),
    "patch_produced": int(patch_bytes) > 0,
    "harness_git_sha": harness_sha,
    "pi_command": [
        "pi", "-p", "<problem_statement>", "--mode", "json",
        "--provider", provider, "--model", served, "--no-session",
    ] + (["--extension", extension] if extension else []),
}
with open(path, "w") as f:
    json.dump(record, f, indent=2)
print(f"[pod] run record -> {path}")
PY

# --- 5. metrics -----------------------------------------------------------
python3 "$HERE/extract_metrics.py" "$OUT" >/dev/null 2>&1 \
  && echo "[pod] metrics -> $OUT/metrics.json" \
  || echo "[pod] metrics extraction skipped (see $OUT/pi_log.jsonl)"

echo
echo "[pod] done -> $OUT"
echo "[pod] verification needs Docker, which RunPod cannot provide."
echo "[pod] copy results to a Docker host, then:"
echo "[pod]   run/verify_task.sh --task $TASK --model $MODEL"
