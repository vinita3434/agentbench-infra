#!/usr/bin/env bash
#
# docker_gc.sh [--aggressive] [--min-free-gb N]
#
# Reclaim Docker disk between tasks. Built after a sweep filled the disk to 98%
# and Docker Desktop refused to start -- at which point `docker prune` is not
# available either, because the daemon is down. Cleaning as you go avoids the
# situation entirely.
#
# What accumulates during a sweep, in order of size:
#   eval images        2-4GB each. verify_*.sh already removes them unless
#                      --keep-image, so this only catches leftovers.
#   dangling images    every rebuild of agent-serving-bench:test orphans the
#                      previous layers. Invisible in `docker images` output.
#   build cache        grew to 7GB in one session.
#   stopped containers small, but they pin image layers so those cannot be freed.
#
# Default is safe: nothing still tagged and in use is touched. --aggressive also
# drops the cached eval images, which cost a re-pull if you verify a task twice.
set -uo pipefail

AGGRESSIVE=0
MIN_FREE_GB=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aggressive) AGGRESSIVE=1; shift ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

free_gb() { df -g /System/Volumes/Data 2>/dev/null | awk 'NR==2{print $4}' \
            || df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'; }

docker info >/dev/null 2>&1 || { echo "[gc] docker not running -- nothing to do" >&2; exit 0; }

before="$(free_gb)"

docker container prune -f >/dev/null 2>&1
# Dangling images only: -a would delete images still tagged and wanted.
docker image prune -f >/dev/null 2>&1
docker builder prune -f >/dev/null 2>&1

if [[ "$AGGRESSIVE" -eq 1 ]]; then
  # Eval images are re-pullable from the registry; results are on the host and
  # are never touched by any of this.
  # Only eval images -- never agent-serving-bench:test, which the next task
  # would immediately have to rebuild from scratch.
  imgs=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
         | grep -E 'polybench|sweb\.eval' || true)
  if [[ -n "$imgs" ]]; then
    echo "$imgs" | xargs -r docker rmi -f >/dev/null 2>&1
    echo "[gc] removed $(wc -l <<<"$imgs" | tr -d ' ') eval image(s)"
  fi
fi

after="$(free_gb)"
echo "[gc] free: ${before}GB -> ${after}GB"

# A sweep that runs out of disk mid-flight loses the episode AND leaves Docker
# unable to start. Better to stop while the results so far are intact.
if [[ "$MIN_FREE_GB" -gt 0 && "$after" -lt "$MIN_FREE_GB" ]]; then
  echo "[gc] WARNING: only ${after}GB free, below the ${MIN_FREE_GB}GB floor." >&2
  echo "[gc] Re-run with --aggressive, or stop the sweep before Docker wedges." >&2
  exit 1
fi
