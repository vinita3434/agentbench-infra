#!/usr/bin/env bash
# status.sh [--model <slug>] [--pending] [--save [path]]
#
# One line per run: what the agent did, and whether it has been scored.
#
# The two are deliberately separate columns. An agent run can succeed (patch
# produced, exit 0) while verification never ran or could not run -- and those
# are different problems with different fixes. A task only counts toward the
# resolve-rate goal when the third column says `resolved`.
#
#   --model    restrict to one model's results tree (default: all)
#   --pending  show only runs that are not yet resolved (the work queue)
#   --save     also write the resolved set to results/resolved_tasks.txt
#
# The saved file is the point of the whole exercise: a task that a frontier
# model resolved AND that the verifier scored is a known-good benchmark item,
# so it can be re-run against open-weight models on any serving stack and the
# resulting number means something. Regenerate it, never hand-edit it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
MODEL_FILTER=""
PENDING=0
SAVE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_FILTER="$2"; shift 2 ;;
    --pending) PENDING=1; shift ;;
    --save)
      # optional path argument; bare --save uses the default
      if [[ $# -ge 2 && "$2" != --* ]]; then SAVE="$2"; shift 2
      else SAVE="$ROOT/results/resolved_tasks.txt"; shift; fi ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT="$ROOT" MODEL_FILTER="$MODEL_FILTER" PENDING="$PENDING" SAVE="$SAVE" python3 - <<'PY'
import json, os, pathlib

root = pathlib.Path(os.environ["ROOT"])
# Must match run_task.sh's slug exactly: `tr '/:' '__'` maps each character to
# a single underscore, so "a/b" -> "a_b" (not "a__b").
mfilter = (os.environ.get("MODEL_FILTER") or "").replace("/", "_").replace(":", "_")
pending_only = os.environ["PENDING"] == "1"

def read(p):
    try:
        return json.load(p.open())
    except (OSError, json.JSONDecodeError):
        return None

manifest = {}
mpath = root / "tasks" / "manifest.jsonl"
if mpath.exists():
    for line in mpath.open():
        try:
            r = json.loads(line)
            manifest[r["instance_id"]] = r
        except (json.JSONDecodeError, KeyError):
            pass

rows, resolved, counted, keepers, hard = [], 0, 0, [], []
for model_dir in sorted((root / "results").iterdir()):
    # _gold holds verifier self-checks, not agent runs; serve/ holds launch records.
    if not model_dir.is_dir() or model_dir.name in {"_gold", "serve", "artifacts"}:
        continue
    if mfilter and model_dir.name != mfilter:
        continue
    for task_dir in sorted(model_dir.iterdir()):
        rr = read(task_dir / "run_record.json")
        if not rr:
            continue
        vv = read(task_dir / "verify.json")
        # metrics.json is written by extract_metrics.py; a run that was never
        # post-processed still shows its record fields, just without turns.
        mm = read(task_dir / "metrics.json") or {}

        if vv is None:
            state, detail = "NOT VERIFIED", "run verify_task.sh"
        elif vv.get("error"):
            # Harness could not score it. NOT a model failure -- kept visually
            # distinct so it never gets averaged in as a zero.
            state, detail = "ERROR", str(vv["error"])[:58]
        else:
            f, p = vv.get("f2p") or {}, vv.get("p2p") or {}
            ok = vv.get("resolved") is True
            state = "resolved" if ok else "not-resolved"
            detail = (f"F2P {f.get('passed','-')}/{f.get('total','-')}  "
                      f"P2P {p.get('passed','-')}/{p.get('total','-')}")
            counted += 1
            resolved += ok
            # Both outcomes are benchmark-ready: what qualifies a task is that
            # verification produced a *verdict*, not that the verdict was good.
            # A task the frontier model failed is the more discriminating item --
            # a set where everything resolves has a ceiling and cannot separate
            # a strong open-weight model from a lucky one.
            if True:
                mf = manifest.get(task_dir.name, {})
                (keepers if ok else hard).append({
                    "task": task_dir.name,
                    "repo": mf.get("repo", rr.get("repo", "")),
                    "language": mf.get("language", ""),
                    "framework": mf.get("test_framework", ""),
                    "verified_with": rr.get("model", ""),
                    "source": vv.get("result_source", ""),
                    "verdict": "solved" if ok else "attempted",
                    "f2p": f.get("total"), "p2p": p.get("total"),
                    "turns": mm.get("turns"), "wall": rr.get("wall_seconds"),
                })

        if pending_only and state == "resolved":
            continue
        rows.append((model_dir.name, task_dir.name, rr.get("wall_seconds", 0),
                     mm.get("turns"), rr.get("patch_bytes", 0), state, detail))

def hms(sec):
    """Runs span 16s to 70min; a bare seconds column stops being readable fast."""
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"

if not rows:
    print("no runs found" + (" (all resolved)" if pending_only else ""))
else:
    print(f"{'task':<34}{'agent':>8}{'turns':>7}{'patch':>8}  {'status':<14}detail")
    print("-" * 116)
    for model, task, wall, turns, patch, state, detail in rows:
        print(f"{task:<34}{hms(wall):>8}{turns if turns is not None else '-':>7}"
              f"{patch:>7}B  {state:<14}{detail}")

# The goal is a count of *verified* resolves, so report that, not a percentage
# of everything attempted -- ERROR rows have no defensible denominator.
print(f"\nresolved {resolved}/{counted} scored"
      f"  ({len(rows)} row{'s' if len(rows) != 1 else ''} shown)")

save = os.environ.get("SAVE")
if save:
    import datetime
    keepers.sort(key=lambda k: k["task"])
    out = pathlib.Path(save)
    out.parent.mkdir(parents=True, exist_ok=True)
    hard.sort(key=lambda k: k["task"])
    with out.open("w") as fh:
        fh.write("# Benchmark-ready SWE-PolyBench tasks\n")
        fh.write("#\n")
        fh.write("# Every task here was *scored* by run/verify_task.sh: the eval image\n")
        fh.write("# exists, the patch applies, and the F2P/P2P tests parse. That is what\n")
        fh.write("# makes it safe to run against open-weight models on any serving stack --\n")
        fh.write("# a 0 is then a model result, not a harness failure. Tasks that could not\n")
        fh.write("# be scored at all (ERROR) are deliberately absent.\n")
        fh.write("#\n")
        fh.write("# Two sections, and you want both. SOLVED items confirm a model can do the\n")
        fh.write("# work; UNSOLVED items are ones a frontier model failed *with the verifier\n")
        fh.write("# working correctly*, so they still discriminate at the top of the range.\n")
        fh.write("# A set of only-solved tasks has a ceiling: every decent model scores the\n")
        fh.write("# same and the benchmark stops telling you anything.\n")
        fh.write("#\n")
        fh.write("# Re-run one (Mode B, SGLang on :30000):\n")
        fh.write("#   MODEL=local-model PROVIDER=selfhosted \\\n")
        fh.write("#     run/run_task.sh --task <id> --mode native --workdir <checkout>\n")
        fh.write("#   MODEL=local-model run/verify_task.sh --task <id>\n")
        fh.write("#\n")
        fh.write(f"# regenerated {datetime.datetime.now():%Y-%m-%d %H:%M} "
                 f"by run/status.sh --save -- do not hand-edit\n")
        fh.write(f"# {len(keepers)} solved + {len(hard)} unsolved "
                 f"= {len(keepers) + len(hard)} benchmark-ready\n")

    def section(fh, title, items, note):
        fh.write(f"\n# ==== {title} ({len(items)}) ====\n")
        fh.write(f"# {note}\n\n")
        for k in items:
            fh.write(f"{k['task']}\n")
            fh.write(f"    repo={k['repo']}  lang={k['language']}  "
                     f"framework={k['framework']}  result_source={k['source']}\n")
            fh.write(f"    f2p={k['f2p']}  p2p={k['p2p']}  "
                     f"{k['verdict']}_by={k['verified_with']}  "
                     f"turns={k['turns']}  wall={k['wall']}s\n\n")

    with out.open("a") as fh:
        section(fh, "SOLVED by a frontier model", keepers,
                "Known achievable. An open-weight model failing these is a real gap.")
        section(fh, "UNSOLVED by a frontier model", hard,
                "Scored cleanly, but the fix was wrong. Headroom -- keeps the set "
                "from topping out.")
    print(f"saved {len(keepers)} solved + {len(hard)} unsolved "
          f"= {len(keepers) + len(hard)} benchmark-ready task(s) -> {save}")
PY
