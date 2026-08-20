#!/usr/bin/env bash
#
# validate_swebench.sh [--tasks FILE] [--limit N] [--keep-image]
#
# Decide which SWE-bench Lite tasks are usable benchmark items. Two checks per
# task, and a task needs BOTH:
#
#   gold control   apply the maintainers' patch -> every F2P must PASS.
#                  Proves the task can be solved and that we can score it.
#   base control   apply NOTHING -> at least one F2P must FAIL.
#                  Proves the task can be failed.
#
# The second check is the one that is easy to forget and expensive to skip.
# psf__requests-863 passes gold cleanly, but its F2P tests also pass at base
# commit with no fix at all -- so a model that edits nothing scores 4/4. Gold
# alone would have waved it through, and every model would "resolve" it.
#
# Tasks whose test ids are not pytest node ids (django, sympy) are skipped:
# they need their own runners and verify_swebench.sh refuses them.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
TASKS="$ROOT/tasks/swebench_lite/manifest.jsonl"
LIMIT=0
KEEP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks) TASKS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --keep-image) KEEP="--keep-image"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPORT="$ROOT/results/swebench_validation.tsv"
mkdir -p "$ROOT/results"
[[ -f "$REPORT" ]] || printf 'task\tgold_f2p\tbase_f2p\tverdict\tnote\n' > "$REPORT"

ids=$(python3 -c "
import json, sys
for line in open('$TASKS'):
    r = json.loads(line)
    print(r['instance_id'])
")

n=0
for task in $ids; do
  [[ "$LIMIT" -gt 0 && "$n" -ge "$LIMIT" ]] && break
  grep -q "^${task}	" "$REPORT" && { echo "[validate] $task already checked"; continue; }

  # Skip repos with non-pytest ids before spending a pull.
  n=$((n + 1))

  echo; echo "=== [validate] $task ==============================="
  "$HERE/verify_swebench.sh" --task "$task" --gold --keep-image >/dev/null 2>&1
  gold=$(python3 -c "
import json
try:
    v = json.load(open('$ROOT/results/_gold_swebench/$task/verify.json'))
    f = v.get('f2p') or {}
    print(f\"{f.get('passed')}/{f.get('total')}\")
except Exception: print('none')")

  # Base control: an empty patch means nothing is fixed.
  base_dir="$ROOT/results/_base_swebench/$task"
  mkdir -p "$base_dir"; : > "$base_dir/candidate.patch"
  "$HERE/verify_swebench.sh" --task "$task" --model _base_swebench $KEEP >/dev/null 2>&1
  base=$(python3 -c "
import json
try:
    v = json.load(open('$base_dir/verify.json'))
    f = v.get('f2p') or {}
    print(f\"{f.get('passed')}/{f.get('total')}\")
except Exception: print('none')")

  verdict=$(python3 -c "
g, b = '$gold', '$base'
def parts(s):
    try:
        a, t = s.split('/'); return int(a), int(t)
    except Exception: return None, None
gp, gt = parts(g); bp, bt = parts(b)
if gp is None or bp is None: print('BROKEN|could not score')
elif gt == 0:                print('BROKEN|no F2P tests')
elif gp != gt:               print('BROKEN|gold does not pass')
elif bp == bt:               print('NON-DISCRIMINATING|F2P pass with no fix')
else:                        print('VALID|')")
  v="${verdict%%|*}"; note="${verdict#*|}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$task" "$gold" "$base" "$v" "$note" >> "$REPORT"
  echo "[validate] $task  gold=$gold  base=$base  -> $v $note"
done

echo; echo "=== summary ==="
awk -F'\t' 'NR>1 {c[$4]++} END {for (k in c) printf "  %-20s %d\n", k, c[k]}' "$REPORT"
echo "  report -> $REPORT"
