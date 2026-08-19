#!/usr/bin/env bash
#
# verify_swebench.sh --task <instance_id> [--model <slug>] [--gold] [--keep-image]
#
# Score a patch against SWE-bench Lite's hidden tests. Sibling to
# verify_task.sh, which serves SWE-PolyBench; the two benchmarks ship different
# contracts and mixing them would let a PolyBench-shaped verifier be pointed at
# a SWE-bench task and fail in a way that looks like a model result.
#
# What differs from PolyBench:
#   image        docker.io/swebench/sweb.eval.x86_64.<id>, with `__` -> `_1776_`
#   test command NOT in the dataset. SWE-bench's official harness derives it per
#                repo. Here we exploit the fact that FAIL_TO_PASS/PASS_TO_PASS
#                are pytest node ids for most repos, so pytest can be handed the
#                ids directly -- no per-repo table needed.
#   env          the repo lives at /testbed with deps in a conda env "testbed"
#
# Repos whose test ids are NOT pytest node ids (django's "test (module.Class)",
# sympy's bare function names) are refused up front rather than mis-run.
#
# Writes results/<model>/<task>/verify.json in the SAME shape verify_task.sh
# produces, so status.sh, peek.sh, master_table.py and the artifact builder all
# work against SWE-bench results with no changes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

TASK=""; USE_GOLD=0; KEEP_IMAGE=0
MODEL="${MODEL:-anthropic/claude-sonnet-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gold) USE_GOLD=1; shift ;;
    --keep-image) KEEP_IMAGE=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" ]] || { echo "usage: verify_swebench.sh --task <instance_id> [--gold]" >&2; exit 2; }

TASK_JSON="$REPO_ROOT/tasks/swebench_lite/data/${TASK}.json"
[[ -f "$TASK_JSON" ]] || { echo "no task file: $TASK_JSON" >&2; exit 1; }

MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"
if [[ "$USE_GOLD" -eq 1 ]]; then
  OUT="$REPO_ROOT/results/_gold_swebench/${TASK}"; PATCH_SOURCE="gold"
else
  OUT="$REPO_ROOT/results/${MODEL_SLUG}/${TASK}"; PATCH_SOURCE="candidate"
fi
mkdir -p "$OUT"

IMAGE="docker.io/swebench/sweb.eval.x86_64.$(printf '%s' "$TASK" | sed 's/__/_1776_/'):latest"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- stage patches + the node ids to run ----------------------------------
python3 - "$TASK_JSON" "$TMP" "$USE_GOLD" "$OUT/candidate.patch" <<'PY'
import json, os, sys
task_json, tmp, use_gold, candidate_path = sys.argv[1:5]
task = json.load(open(task_json))

def as_list(raw):
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []

f2p, p2p = as_list(task.get("FAIL_TO_PASS")), as_list(task.get("PASS_TO_PASS"))

# Address tests by FILE, not by node id. Two reasons, both learned the hard
# way: parametrized ids carry ':' and quotes that do not survive as command
# line arguments (matplotlib-23314 failed with "ERROR: not found" on a gold
# patch), and some tasks carry parsing debris in PASS_TO_PASS ("[100%]") that
# is not a test id at all. Running the files and matching names out of the
# JUnit xml sidesteps both -- it costs extra test time, never correctness.
addressable = [t for t in (f2p + p2p) if "::" in t]
if not addressable:
    sys.exit("no pytest-style test ids for this task (django/sympy use their "
             "own runners; not handled here).")
if not any("::" in t for t in f2p):
    sys.exit("no addressable FAIL_TO_PASS ids -- cannot score this task.")
test_files = sorted({t.split("::", 1)[0] for t in addressable})

if use_gold == "1":
    model_patch = task.get("patch") or ""
else:
    if not os.path.exists(candidate_path):
        sys.exit(f"no candidate patch at {candidate_path} -- run the agent first, or use --gold")
    model_patch = open(candidate_path).read()

open(os.path.join(tmp, "model_patch.diff"), "w").write(model_patch)
open(os.path.join(tmp, "test_patch.diff"), "w").write(task.get("test_patch") or "")
# Node ids, one per line, NUL-safe for xargs in the runner.
open(os.path.join(tmp, "node_ids.txt"), "w").write("".join(f"{t}\n" for t in test_files))

# Files the test patch touches: reset before applying it, so an agent that
# wrote its own fixture cannot block the graded tests.
paths = set()
for line in (task.get("test_patch") or "").splitlines():
    if line.startswith(("+++ ", "--- ")):
        p = line[4:].strip().split("\t")[0]
        if p not in ("/dev/null", ""):
            paths.add(p[2:] if p[:2] in ("a/", "b/") else p)
open(os.path.join(tmp, "test_paths.txt"), "w").write("".join(f"{p}\n" for p in sorted(paths)))

runner = f"""#!/bin/bash
source /opt/miniconda3/bin/activate testbed 2>/dev/null || true
cd /testbed || exit 3
git config --global --add safe.directory /testbed 2>/dev/null || true
git reset --hard >/dev/null 2>&1 || true
git clean -fd >/dev/null 2>&1 || true
git checkout {task['base_commit']} >/dev/null 2>&1 || true

if git apply --whitespace=nowarn /verify/model_patch.diff >/verify/apply_model.log 2>&1; then
  echo OK > /verify/model_apply.status
else
  patch -p1 --batch --fuzz=5 < /verify/model_patch.diff >>/verify/apply_model.log 2>&1 \
    && echo OK > /verify/model_apply.status || echo FAIL > /verify/model_apply.status
fi

# The graded tests win over anything the agent wrote at the same path.
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
  patch -p1 --batch --fuzz=5 < /verify/test_patch.diff >>/verify/apply_test.log 2>&1 \
    && echo OK > /verify/test_apply.status || echo FAIL > /verify/test_apply.status
fi

# Run whole test files; the parser matches F2P/P2P by name from the xml.
tr '\\n' '\\0' < /verify/node_ids.txt | xargs -0 --no-run-if-empty \
  python -m pytest -rA -p no:cacheprovider --continue-on-collection-errors \
    --junitxml=/verify/test-results.xml \
    > /verify/test_stdout.txt 2>/verify/test_stderr.txt
echo "exit=$?" > /verify/test_command.status
"""
open(os.path.join(tmp, "run.sh"), "w").write(runner)
os.chmod(os.path.join(tmp, "run.sh"), 0o755)
print(f"[swebench] staged {'gold' if use_gold=='1' else 'candidate'} patch + "
      f"{len(f2p)} F2P / {len(p2p)} P2P across {len(test_files)} test file(s)")
PY

# --- pull + run -----------------------------------------------------------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[swebench] pulling $IMAGE (large, x86 -- first time is slow)..."
  docker pull --platform linux/amd64 -q "$IMAGE" >/dev/null
fi

echo "[swebench] running tests in /testbed ..."
docker run --rm --platform linux/amd64 -v "$TMP":/verify "$IMAGE" bash /verify/run.sh \
  >/dev/null 2>&1 || true

MODEL_APPLY="$(cat "$TMP/model_apply.status" 2>/dev/null || echo FAIL)"
TEST_APPLY="$(cat "$TMP/test_apply.status" 2>/dev/null || echo FAIL)"
for f in test_stdout.txt test_stderr.txt test-results.xml apply_model.log apply_test.log test_reset.log; do
  cp "$TMP/$f" "$OUT/$f" 2>/dev/null || true
done

# --- parse ----------------------------------------------------------------
# Reuses the PolyBench parser: FAIL_TO_PASS/PASS_TO_PASS are renamed to F2P/P2P
# in a temp view so one parser serves both benchmarks.
VIEW="$TMP/task_view.json"
python3 - "$TASK_JSON" "$VIEW" <<'PY'
import json, sys
src, dst = sys.argv[1:3]
t = json.load(open(src))
t["F2P"], t["P2P"] = t.get("FAIL_TO_PASS"), t.get("PASS_TO_PASS")
json.dump(t, open(dst, "w"))
PY

python3 "$HERE/_verify_parse.py" \
  "$VIEW" "$TMP/test_stdout.txt" "$OUT/verify.json" \
  "$PATCH_SOURCE" "$MODEL_APPLY" "$TEST_APPLY" \
  "$TMP/test_stderr.txt" "$TMP/test-results.xml"

echo "[swebench] wrote $OUT/verify.json"
if [[ "$KEEP_IMAGE" -eq 0 ]]; then
  docker rmi "$IMAGE" >/dev/null 2>&1 && echo "[swebench] removed image (use --keep-image to keep)" || true
fi
