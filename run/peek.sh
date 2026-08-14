#!/usr/bin/env bash
# peek.sh --task <instance_id> [--model <slug>]
#
# A cheap read on a finished agent run, before paying for verification.
#
# This is a HINT, never a verdict. The only thing that decides resolved/not is
# run/verify_task.sh running the hidden tests. What this does is compare the
# candidate patch against the dataset's gold patch -- which files each touches,
# how many lines each changes -- because "edited the same file as gold" has
# tracked the outcome closely so far, and "produced no patch at all" is a
# guaranteed failure worth catching before a multi-GB image pull.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
TASK=""
MODEL="${MODEL:-anthropic/claude-sonnet-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" ]] || { echo "usage: peek.sh --task <instance_id> [--model <slug>]" >&2; exit 2; }

MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"
ROOT="$ROOT" TASK="$TASK" DIR="$ROOT/results/${MODEL_SLUG}/${TASK}" python3 - <<'PY'
import json, os, pathlib, re

root = pathlib.Path(os.environ["ROOT"])
task = os.environ["TASK"]
d = pathlib.Path(os.environ["DIR"])

if not (d / "run_record.json").exists():
    raise SystemExit(f"no run found at {d}")

rr = json.load((d / "run_record.json").open())
mm = {}
if (d / "metrics.json").exists():
    mm = json.load((d / "metrics.json").open())
gold = json.load((root / "tasks/data" / f"{task}.json").open()).get("patch") or ""
cand = (d / "candidate.patch").read_text() if (d / "candidate.patch").exists() else ""

TEST_RE = re.compile(r"(^|/)(tests?|__tests__|specs?)(/|$)")


def files_of(diff):
    out = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            bits = line.split()
            if len(bits) >= 4 and bits[-1].startswith("b/"):
                out.append(bits[-1][2:])
    return out


def churn(diff):
    add = sum(1 for l in diff.splitlines()
              if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in diff.splitlines()
              if l.startswith("-") and not l.startswith("---"))
    return add, rem


cf, gf = files_of(cand), files_of(gold)
src_cf = [f for f in cf if not TEST_RE.search(f)]
test_cf = [f for f in cf if TEST_RE.search(f)]
ca, cr = churn(cand)
ga, gr = churn(gold)

wall = rr.get("wall_seconds", 0)
m, s = divmod(int(wall), 60)
overlap = sorted(set(src_cf) & set(gf))

print(f"\n{task}   [{rr.get('model', '?')}]")
print(f"  patch      {rr.get('patch_bytes', 0)} B, {len(cf)} file(s), +{ca}/-{cr} lines")
print(f"  agent      {mm.get('turns', '?')} turns, {m}m{s:02d}s")
if test_cf:
    print(f"  test files {len(test_cf)} touched -> stripped at verify: "
          f"{', '.join(test_cf[:3])}")
print(f"  gold       {len(gf)} file(s), +{ga}/-{gr} lines: {', '.join(gf[:3])}")
print(f"  overlap    {len(overlap)}/{len(gf)} of gold's files also edited"
      + (f": {', '.join(overlap[:3])}" if overlap else ""))

# Ranked most- to least-informative. Only the first and last are near-certain;
# the middle is genuinely a coin flip and is labelled as such.
if not cand.strip():
    verdict, why = "WILL FAIL", "no patch produced -- nothing to score"
elif not src_cf:
    verdict, why = "WILL FAIL", "only test files edited; no source change survives stripping"
elif cand.strip() == gold.strip():
    verdict, why = "ALMOST CERTAIN", "candidate is byte-identical to gold"
elif set(src_cf) == set(gf):
    verdict, why = "LIKELY", "edited exactly the files gold edits"
elif overlap:
    verdict, why = "PLAUSIBLE", "edited some of gold's files, plus others"
else:
    verdict, why = "UNLIKELY", "touched none of the files gold changes"

print(f"\n  guess      {verdict}  ({why})")
print("  NOT a verdict -- only run/verify_task.sh decides resolved/not.\n")
PY
