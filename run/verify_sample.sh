#!/usr/bin/env bash
# verify_sample.sh [--per-family N] [--only <repo/name>] [--dry-run]
#
# Prove the verifier works on the current task set, without paying for all of it.
#
# Every verification failure seen so far has been a property of the *family*
# (repo + test framework) rather than of the individual task -- a TensorFlow
# suite SIGILLs under x86 emulation for every task in it, a maven suite that
# writes no surefire XML writes none for any of them. So grading the dataset's
# gold patch for one task per family generalizes to the family: if gold cannot
# be scored, no model's patch can be either.
#
# Reads tasks/manifest.jsonl, picks the N simplest tasks per (repo, framework),
# and runs verify_task.sh --gold on each. Results land in results/_gold/<id>/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PER_FAMILY=1
ONLY=""
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --per-family) PER_FAMILY="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DRY" -eq 0 ]]; then
  command -v docker >/dev/null || { echo "[sample] docker not found" >&2; exit 1; }
  docker info >/dev/null 2>&1 || { echo "[sample] docker daemon is not running" >&2; exit 1; }
fi

TASKS=$(ROOT="$ROOT" PER_FAMILY="$PER_FAMILY" ONLY="$ONLY" python3 - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["ROOT"])
per = int(os.environ["PER_FAMILY"])
only = os.environ.get("ONLY") or None
rows = [json.loads(l) for l in (root / "tasks/manifest.jsonl").open()]
if only:
    rows = [r for r in rows if r["repo"] == only]
# Manifest is already written simplest-first, so first-seen per family is the
# cheapest task to grade in it.
seen = {}
picked = []
for r in rows:
    key = (r["repo"], r["test_framework"])
    if seen.get(key, 0) < per:
        seen[key] = seen.get(key, 0) + 1
        picked.append(r["instance_id"])
print("\n".join(picked))
PY
)

[[ -n "$TASKS" ]] || { echo "[sample] no tasks matched" >&2; exit 1; }
echo "[sample] $(wc -w <<<"$TASKS") task(s) to gold-verify:"
printf '  %s\n' $TASKS
[[ "$DRY" -eq 1 ]] && exit 0

pass=(); fail=()
for t in $TASKS; do
  echo; echo "=== [sample] gold verify $t ==============================="
  # Keep going on failure: a broken family is a *result*, not an abort.
  "$HERE/verify_task.sh" --task "$t" --gold || true
  v="$ROOT/results/_gold/$t/verify.json"
  if [[ -f "$v" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$v')).get('resolved') is True else 1)"; then
    pass+=("$t")
  else
    fail+=("$t")
  fi
done

echo
echo "=== [sample] summary ======================================"
for t in "${pass[@]:-}"; do [[ -n "$t" ]] && echo "  PASS  $t"; done
for t in "${fail[@]:-}"; do [[ -n "$t" ]] && echo "  FAIL  $t   <- family not gradeable; update tasks/verifiability.py"; done
[[ ${#fail[@]} -eq 0 ]]
