#!/usr/bin/env python3
"""Show what an agent actually did, turn by turn, for one or every attempt.

    python run/trajectory.py --task <id> --model <slug>            # all attempts
    python run/trajectory.py --task <id> --model <slug> --attempt 2
    python run/trajectory.py --task <id> --model <slug> --compare  # side by side

Attempts live in sibling directories: results/<model>/<task>/ is the latest and
results/<model>/<task>__aN/ are archived earlier ones, ordered by run timestamp.
Retries of the same task by the same model are the point of this view -- nothing
about the task changed between them, so any difference in the trajectory is the
model taking a different path, which is exactly what a "flaky" failure looks
like versus a systematic one.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def attempts_for(model_slug: str, task: str):
    """Every attempt directory for (model, task), oldest first."""
    base = RESULTS / model_slug
    if not base.is_dir():
        return []
    dirs = [d for d in base.iterdir()
            if d.is_dir() and (d.name == task or d.name.startswith(f"{task}__"))
            and (d / "run_record.json").exists()]
    # Timestamp order, not directory name -- the un-suffixed dir is the newest.
    return sorted(dirs, key=lambda d: json.loads(
        (d / "run_record.json").read_text()).get("timestamp_utc", ""))


def summarise(directory: pathlib.Path):
    rec = json.loads((directory / "run_record.json").read_text())
    metrics = {}
    mp = directory / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    verify = None
    vp = directory / "verify.json"
    if vp.exists():
        verify = json.loads(vp.read_text())

    cost, calls = 0.0, 0
    turns = []
    turn = 0
    log = directory / "pi_log.jsonl"
    if log.exists():
        for line in log.open():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "turn_start":
                turn += 1
            elif kind == "tool_execution_start":
                args = event.get("args") or {}
                detail = (args.get("command") or args.get("path")
                          or json.dumps(args, default=str))
                turns.append((turn, event.get("toolName") or "?", str(detail)))
            elif kind == "message_end":
                usage = (event.get("message") or {}).get("usage") or {}
                c = (usage.get("cost") or {}).get("total")
                if c is not None:
                    cost += c
                    calls += 1
    return rec, metrics, verify, turns, (cost if calls else None)


def verdict(verify, rec):
    if (rec.get("patch_bytes") or 0) == 0 and (rec.get("wall_seconds") or 99) < 10:
        return "402 (never ran)"
    if verify is None:
        return "not verified"
    if verify.get("error"):
        return "ERROR"
    f = verify.get("f2p") or {}
    p = verify.get("p2p") or {}
    tag = "PASS" if verify.get("resolved") else "FAIL"
    return f"{tag}  F2P {f.get('passed')}/{f.get('total')}  P2P {p.get('passed')}/{p.get('total')}"


def show(directory, index, total, width=96, brief=False):
    rec, metrics, verify, turns, cost = summarise(directory)
    mins, secs = divmod(int(rec.get("wall_seconds") or 0), 60)
    print(f"\n{'=' * width}")
    print(f"ATTEMPT {index}/{total}   {rec.get('task_id')}   {rec.get('model')}")
    print(f"  {verdict(verify, rec)}")
    print(f"  {metrics.get('turns', '?')} turns · {mins}m{secs:02d}s · "
          f"{'$%.4f' % cost if cost is not None else 'cost n/a'} · "
          f"patch {rec.get('patch_bytes')}B · {rec.get('timestamp_utc')}")
    if brief:
        return
    print(f"{'-' * width}")
    if not turns:
        print("  (no tool calls -- the agent never acted)")
        return
    for t, tool, detail in turns:
        detail = detail.replace("\n", " ")[:width - 18]
        print(f"  t{t:<4}{tool:<8}{detail}")

    # Tool mix says a lot: bash-heavy with no edit is the shape of an agent
    # that explored and never committed to a change.
    counts = {}
    for _, tool, _ in turns:
        counts[tool] = counts.get(tool, 0) + 1
    print(f"  {'-' * (width - 2)}")
    print("  tools: " + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--attempt", type=int, help="1-based; default shows all")
    ap.add_argument("--compare", action="store_true",
                    help="summaries only, for scanning many attempts at once")
    args = ap.parse_args()

    slug = args.model.replace("/", "_").replace(":", "_")
    dirs = attempts_for(slug, args.task)
    if not dirs:
        sys.exit(f"no runs found for {args.task} under results/{slug}/")

    if args.attempt:
        if not 1 <= args.attempt <= len(dirs):
            sys.exit(f"attempt {args.attempt} out of range (1..{len(dirs)})")
        dirs = [dirs[args.attempt - 1]]
        show(dirs[0], args.attempt, len(dirs), brief=args.compare)
    else:
        for i, d in enumerate(dirs, 1):
            show(d, i, len(dirs), brief=args.compare)
    print()


if __name__ == "__main__":
    main()
