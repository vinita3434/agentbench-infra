#!/usr/bin/env bash
#
# verify_task.sh --task <instance_id> [--model <slug>] [--gold] [--keep-image]
#
# Score a candidate patch against a task's hidden F2P/P2P tests, using
# SWE-PolyBench's pre-built per-instance image on GHCR (which already has
# /testbed checked out at base_commit with deps installed).
#
#   1. Take the patch to grade: the model's results/<model>/<task>/candidate.patch
#      (default) or the dataset gold `patch` (--gold, for validating this script).
#   2. Pull ghcr.io/timesler/swe-polybench.eval.x86_64.<instance_id>:v1.1.
#   3. In /testbed: git apply <patch>, then git apply <test_patch> (adds the
#      F2P/P2P tests the agent never saw), then run the task's test_command
#      (which ends in jest --json).
#   4. Parse per-test results -> results/.../verify.json:
#        F2P x/N (%), P2P y/M (%), resolved = ALL F2P pass AND ALL P2P pass.
#
# --gold needs no API key and no prior run: it proves the verifier is correct
# (gold patch should make F2P go green).
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

TASK=""
MODEL="${MODEL:-anthropic/claude-sonnet-5}"
USE_GOLD=0
KEEP_IMAGE=0
IMAGE_OVERRIDE=""
IMAGE_TAG="${POLYBENCH_IMAGE_TAG:-latest}"   # all instances have 'latest' and 'v1.0'
TASK_FILE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --task-file) TASK_FILE_OVERRIDE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gold) USE_GOLD=1; shift ;;
    --keep-image) KEEP_IMAGE=1; shift ;;
    --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
    --tag) IMAGE_TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" ]] || { echo "usage: verify_task.sh --task <id> [--gold]" >&2; exit 2; }

# --task-file lets a caller score an instance that is not in the committed task
# set (e.g. probing candidates) without writing into tasks/data.
TASK_JSON="${TASK_FILE_OVERRIDE:-$REPO_ROOT/tasks/data/${TASK}.json}"
[[ -f "$TASK_JSON" ]] || { echo "no task file: $TASK_JSON" >&2; exit 1; }

MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"
if [[ "$USE_GOLD" -eq 1 ]]; then
  OUT="$REPO_ROOT/results/_gold/${TASK}"; PATCH_SOURCE="gold"
else
  OUT="$REPO_ROOT/results/${MODEL_SLUG}/${TASK}"; PATCH_SOURCE="candidate"
fi
mkdir -p "$OUT"

IMAGE="${IMAGE_OVERRIDE:-ghcr.io/timesler/swe-polybench.eval.x86_64.${TASK}:${IMAGE_TAG}}"

# --- stage patches + runner into a temp dir mounted into the container ----
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Write model/gold patch, test patch, and the runner script via python so the
# (large, special-char-laden) test_command needs no shell escaping.
python3 - "$TASK_JSON" "$TMP" "$USE_GOLD" "$OUT/candidate.patch" <<'PY'
import json, os, sys
task_json, tmp, use_gold, candidate_path = sys.argv[1:5]
task = json.load(open(task_json))

if use_gold == "1":
    model_patch = task.get("patch") or ""
else:
    if not os.path.exists(candidate_path):
        sys.exit(f"no candidate patch at {candidate_path} -- run run_task.sh first, or use --gold")
    model_patch = open(candidate_path).read()

TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "testing"}


def is_test_path(path):
    """Heuristic: does this file belong to the graded test surface?

    Deliberately broad. A false positive drops a change the agent probably
    should not have made anyway; a false negative lets an agent-authored test
    into a glob-based suite (svelte's css runner picks up every directory under
    test/css/samples/), where it silently becomes an extra graded case.
    """
    parts = path.split("/")
    if TEST_DIRS.intersection(parts):
        return True
    base = parts[-1]
    if base in ("conftest.py",):
        return True
    stem = base.rsplit(".", 1)[0]
    return (".test." in base or ".spec." in base
            or stem.startswith("test_") or stem.endswith("_test"))


def split_diff(text):
    """Unified diff -> [(path, section_text)], one entry per file."""
    out, path, buf = [], None, []
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if buf:
                out.append((path, "".join(buf)))
            buf = [line]
            bits = line.split()
            path = bits[-1][2:] if len(bits) >= 4 and bits[-1][:2] == "b/" else None
        else:
            buf.append(line)
    if buf:
        out.append((path, "".join(buf)))
    return out


dropped = []
if use_gold != "1" and model_patch.strip():
    # The agent may write tests -- that is normal engineering behaviour and we
    # do not forbid it. It just must not reach the grader: the hidden tests are
    # the measurement instrument, so anything the agent wrote on that surface is
    # removed before its patch is applied, and recorded rather than hidden.
    kept = []
    for p, section in split_diff(model_patch):
        if p and is_test_path(p):
            dropped.append(p)
        else:
            kept.append(section)
    if dropped:
        model_patch = "".join(kept)
open(os.path.join(tmp, "test_surface_dropped.txt"),
     "w").write("".join(f"{p}\n" for p in dropped))

open(os.path.join(tmp, "model_patch.diff"), "w").write(model_patch)
test_patch = task.get("test_patch") or ""
open(os.path.join(tmp, "test_patch.diff"), "w").write(test_patch)

# Every path the test patch touches. The graded tests are authoritative, so the
# runner resets these to base state before applying it -- otherwise an agent
# that writes its own fixture at one of these paths blocks `git apply` and the
# task scores not-resolved for a reason unrelated to its fix.
test_paths = set()
for line in test_patch.splitlines():
    if line.startswith("+++ ") or line.startswith("--- "):
        p = line[4:].strip().split("\t")[0]
        if p in ("/dev/null", ""):
            continue
        if p[:2] in ("a/", "b/"):
            p = p[2:]
        test_paths.add(p)
# Trailing newline is required: `while read` drops a final unterminated line,
# which would silently leave one path un-reset.
open(os.path.join(tmp, "test_paths.txt"), "w").write(
    "".join(f"{p}\n" for p in sorted(test_paths)))

test_command = task.get("test_command") or ""
runner = f"""#!/bin/bash
cd /testbed || exit 3
# clean any stray state, keep base_commit tree
git reset --hard >/dev/null 2>&1 || true
git clean -fd  >/dev/null 2>&1 || true

if git apply --whitespace=nowarn /verify/model_patch.diff >/verify/apply_model.log 2>&1; then
  echo OK > /verify/model_apply.status
else
  patch -p1 < /verify/model_patch.diff >>/verify/apply_model.log 2>&1 \
    && echo OK > /verify/model_apply.status || echo FAIL > /verify/model_apply.status
fi

# The graded tests win over anything the agent wrote at the same path: restore
# tracked files, delete ones the agent created. Logged, not silent -- a reset
# here means the agent touched the test surface, which is worth seeing.
: > /verify/test_reset.log
while IFS= read -r p; do
  [ -n "$p" ] || continue
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
    git checkout HEAD -- "$p" 2>/dev/null && echo "restored $p" >> /verify/test_reset.log
  elif [ -e "$p" ]; then
    rm -rf "$p" && echo "removed  $p" >> /verify/test_reset.log
  fi
done < /verify/test_paths.txt

if git apply --whitespace=nowarn /verify/test_patch.diff >/verify/apply_test.log 2>&1; then
  echo OK > /verify/test_apply.status
else
  patch -p1 < /verify/test_patch.diff >>/verify/apply_test.log 2>&1 \
    && echo OK > /verify/test_apply.status || echo FAIL > /verify/test_apply.status
fi

{{ {test_command} ; }} > /verify/test_stdout.txt 2>/verify/test_stderr.txt
echo "exit=$?" > /verify/test_command.status
# pytest/surefire write a JUnit XML (e.g. --junitxml=test-results.xml in /testbed).
cp /testbed/test-results.xml /verify/test-results.xml 2>/dev/null || true
# pytest --json-report writes its results to a file, never to stdout, and the
# filename is task-configurable (--json-report-file=...), so .report.json is
# only the default. Take the real path from the line pytest prints, and fall
# back to the two common names if that line is absent.
REPORT_REL="$(sed -n 's/^report saved to: //p' /verify/test_stdout.txt | tail -1)"
if [ -n "$REPORT_REL" ]; then
  cp "/testbed/$REPORT_REL" /verify/report.json 2>/dev/null \
    || cp "$REPORT_REL" /verify/report.json 2>/dev/null || true
fi
[ -f /verify/report.json ] || cp /testbed/.report.json /verify/report.json 2>/dev/null || true
[ -f /verify/report.json ] || cp /testbed/report.json  /verify/report.json 2>/dev/null || true
# Maven surefire scatters per-class XML under target/surefire-reports/.
find /testbed -path '*/surefire-reports/TEST-*.xml' -exec cp {{}} /verify/ \\; 2>/dev/null || true
"""
open(os.path.join(tmp, "run.sh"), "w").write(runner)
os.chmod(os.path.join(tmp, "run.sh"), 0o755)
print(f"[verify] staged patches ({'gold' if use_gold=='1' else 'candidate'}) + runner in {tmp}")
PY

# --- pull image (skip if already local) -----------------------------------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[verify] pulling $IMAGE (large, x86 -- first time is slow)..."
  docker pull --platform linux/amd64 "$IMAGE"
else
  echo "[verify] image already present: $IMAGE"
fi

# --- run tests in the instance image --------------------------------------
echo "[verify] running test_command in /testbed ..."
docker run --rm --platform linux/amd64 -v "$TMP":/verify "$IMAGE" bash /verify/run.sh \
  || echo "[verify] (container returned non-zero; test failures are expected to be captured)"

MODEL_APPLY="$(cat "$TMP/model_apply.status" 2>/dev/null || echo FAIL)"
TEST_APPLY="$(cat "$TMP/test_apply.status" 2>/dev/null || echo FAIL)"

# keep raw logs next to the result for debugging
cp "$TMP/test_stdout.txt" "$OUT/test_stdout.txt" 2>/dev/null || true
cp "$TMP/test_stderr.txt" "$OUT/test_stderr.txt" 2>/dev/null || true
cp "$TMP/test-results.xml" "$OUT/test-results.xml" 2>/dev/null || true
cp "$TMP/report.json" "$OUT/report.json" 2>/dev/null || true
cp "$TMP/apply_model.log" "$OUT/apply_model.log" 2>/dev/null || true
cp "$TMP/test_reset.log" "$OUT/test_reset.log" 2>/dev/null || true
# Keep the apply logs: a FAIL here is the difference between "model was wrong"
# and "the patch never made it in", and the log is the only place that says which.
cp "$TMP/apply_test.log" "$OUT/apply_test.log" 2>/dev/null || true
cp "$TMP/test_surface_dropped.txt" "$OUT/test_surface_dropped.txt" 2>/dev/null || true
if [[ -s "$TMP/test_surface_dropped.txt" ]]; then
  echo "[verify] dropped $(wc -l < "$TMP/test_surface_dropped.txt" | tr -d ' ') agent edit(s) on the test surface before applying:"
  sed 's/^/[verify]   /' "$TMP/test_surface_dropped.txt"
fi
if [[ -s "$TMP/test_reset.log" ]]; then
  echo "[verify] agent had written $(wc -l < "$TMP/test_reset.log" | tr -d ' ') file(s) on the graded test surface; reset before applying test_patch:"
  sed 's/^/[verify]   /' "$TMP/test_reset.log"
fi

# --- parse into verify.json --------------------------------------------------
# Four possible result sources, in the order the parser tries them: jest json on
# stdout, mocha json on stdout, JUnit xml, pytest-json-report. The last is why
# this passes a 9th argument -- those tasks print results to no file we'd
# otherwise collect.
python3 "$HERE/_verify_parse.py" \
  "$TASK_JSON" "$TMP/test_stdout.txt" "$OUT/verify.json" \
  "$PATCH_SOURCE" "$MODEL_APPLY" "$TEST_APPLY" \
  "$TMP/test_stderr.txt" "$TMP/test-results.xml" "$TMP/report.json"

# Record that the agent reached the test surface. Kept as a flag on the result
# rather than a side note in a log: it is a property of the trajectory, and a
# resolve rate is easier to trust when you can see which runs needed a cleanup.
python3 - "$OUT/verify.json" "$TMP/test_surface_dropped.txt" "$TMP/test_reset.log" <<'PY'
import json, os, sys
verify_path, dropped_path, reset_path = sys.argv[1:4]

def lines(p):
    if not (p and os.path.exists(p)):
        return []
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]

dropped = lines(dropped_path)
reset = lines(reset_path)
try:
    with open(verify_path) as f:
        v = json.load(f)
except (OSError, json.JSONDecodeError):
    sys.exit(0)
v["touched_test_surface"] = bool(dropped or reset)
v["test_surface_dropped"] = dropped
v["test_surface_reset"] = reset
with open(verify_path, "w") as f:
    json.dump(v, f, indent=2)
PY

echo "[verify] wrote $OUT/verify.json"

if [[ "$KEEP_IMAGE" -eq 0 && -z "$IMAGE_OVERRIDE" ]]; then
  docker rmi "$IMAGE" >/dev/null 2>&1 && echo "[verify] removed image $IMAGE (use --keep-image to keep)" || true
fi
