#!/usr/bin/env bash
#
# prepare_repos.sh [--dest DIR] [--tasks FILE] [--dry-run]
#
# Stage the task repos a pod needs, before the GPU is running.
#
# Each task pins its own base_commit, and the 12-task set spans 12 distinct
# commits across 4 repos. Rather than full clones (gigabytes of history nobody
# reads), this fetches exactly the commits the set names, depth 1 each, into one
# directory per repo. run_pod.sh then checks out whichever commit its task wants.
#
# Deliberately does NOT install test dependencies. Dockerfile.test only clones
# the repo -- the frontier baseline faced a bare checkout and the agent dealt
# with it itself. Pre-installing here would make pod runs easier than the runs
# they are compared against, and the comparison is the whole point.
#
# Run this on a cheap CPU pod. It is network-bound and slow; H100 time is not.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

DEST="${DEST:-/workspace/repos}"
TASKS="$ROOT/results/resolved_tasks.txt"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$TASKS" ]] || { echo "no task list at $TASKS" >&2; exit 1; }

# (repo_url, base_commit, task_id) for every task in the list, from tasks/data.
PLAN="$(ROOT="$ROOT" TASKS="$TASKS" python3 - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["ROOT"])
ids = [l.strip() for l in open(os.environ["TASKS"])
       if l.strip() and not l.startswith(("#", " ", "\t"))]
for i in ids:
    p = root / "tasks" / "data" / f"{i}.json"
    if not p.exists():
        continue
    d = json.load(p.open())
    print(f"{d['repo']}\t{d['base_commit']}\t{i}")
PY
)"

[[ -n "$PLAN" ]] || { echo "no tasks resolved from $TASKS" >&2; exit 1; }

echo "[repos] destination: $DEST"
echo "[repos] $(wc -l <<<"$PLAN") task(s):"
echo "$PLAN" | awk -F'\t' '{printf "    %-24s %s  %s\n", $1, substr($2,1,12), $3}'

if [[ "$DRY" -eq 1 ]]; then
  echo "[repos] dry run, nothing fetched"
  exit 0
fi

mkdir -p "$DEST"
df -h "$DEST" | tail -1

fail=0
# One directory per repo; commits accumulate into it.
for repo in $(echo "$PLAN" | cut -f1 | sort -u); do
  name="$(basename "$repo")"
  dir="$DEST/$name"
  url="https://github.com/${repo}.git"

  if [[ ! -d "$dir/.git" ]]; then
    echo
    echo "[repos] init $name  <- $url"
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  # safe.directory is set --local so concurrent workers never race ~/.gitconfig
  git -C "$dir" config --local --add safe.directory "$dir" 2>/dev/null || true
  git -C "$dir" config --local advice.detachedHead false 2>/dev/null || true

  for sha in $(echo "$PLAN" | awk -F'\t' -v r="$repo" '$1==r {print $2}' | sort -u); do
    if git -C "$dir" cat-file -e "${sha}^{commit}" 2>/dev/null; then
      echo "[repos]   $name $sha already present"
      continue
    fi
    echo "[repos]   fetching $name $sha (depth 1)"
    if ! git -C "$dir" fetch -q --depth 1 origin "$sha"; then
      echo "[repos]   WARNING: fetch failed for $name $sha" >&2
      fail=$((fail + 1))
      continue
    fi
  done

  # Leave the tree on one of its commits so the directory is usable as-is.
  first="$(echo "$PLAN" | awk -F'\t' -v r="$repo" '$1==r {print $2; exit}')"
  git -C "$dir" checkout -q "$first" 2>/dev/null || true
done

echo
df -h "$DEST" | tail -1
echo "[repos] $(du -sh "$DEST" 2>/dev/null | cut -f1) used by repos"

# --- verify every commit the set needs is actually present ----------------
echo
echo "[repos] verifying:"
missing=0
while IFS=$'\t' read -r repo sha task; do
  dir="$DEST/$(basename "$repo")"
  if git -C "$dir" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    printf "    ok      %-34s %s\n" "$task" "${sha:0:12}"
  else
    printf "    MISSING %-34s %s\n" "$task" "${sha:0:12}"
    missing=$((missing + 1))
  fi
done <<<"$PLAN"

echo
if [[ "$missing" -gt 0 ]]; then
  echo "[repos] $missing commit(s) missing -- those tasks cannot run" >&2
  exit 1
fi
echo "[repos] all commits present. workdir paths for run_pod.sh:"
for repo in $(echo "$PLAN" | cut -f1 | sort -u); do
  echo "    $repo -> $DEST/$(basename "$repo")"
done
