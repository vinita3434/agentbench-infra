#!/usr/bin/env python3
"""Split an episode's wall time into model time vs tool time.

    python run/timing.py --model anthropic/claude-sonnet-5          # all runs
    python run/timing.py --task <id> --model <slug> --per-turn      # one episode

Wall time is not one thing. An agentic episode alternates between waiting on the
model and running commands locally, and the two behave completely differently:
model time scales with context length and is what a serving stack governs, while
tool time is npm installs and test suites and is unaffected by the model at all.
Reporting only the total hides which one an episode actually spent its life in.

Measured from events.timed.jsonl, which carries a capture-time offset (`_t`) on
every event because Pi's own events are untimestamped:

  model time   assistant message_start -> message_end
  tool  time   tool_execution_start    -> tool_execution_end
  other        whatever is left: process startup, git reset, patch extraction

For OpenRouter runs "model time" is inference PLUS network and queueing -- it is
latency, not GPU occupancy. On a self-hosted endpoint the same measurement is
much closer to real GPU time, which is why the split is worth having now: it is
the baseline the H100 numbers get compared against.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def split_time(directory: pathlib.Path):
    """(model_s, tool_s, turns, per_turn) from the timestamped event stream."""
    path = directory / "events.timed.jsonl"
    if not path.exists():
        return None

    span = 0.0
    model_s = tool_s = 0.0
    per_turn = []
    turn = 0
    msg_open = None          # (start_t, role)
    tool_open = None
    cur = {"model": 0.0, "tool": 0.0}

    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = row.get("_t")
        event = row.get("event") or {}
        kind = event.get("type")
        if t is None:
            continue
        span = max(span, t)

        if kind == "turn_start":
            if turn:
                per_turn.append((turn, cur["model"], cur["tool"]))
            turn += 1
            cur = {"model": 0.0, "tool": 0.0}
        elif kind == "message_start":
            msg_open = (t, ((event.get("message") or {}).get("role")))
        elif kind == "message_end":
            if msg_open:
                start, role = msg_open
                # Only assistant messages cost model time; the user turn is
                # just the harness handing over the prompt.
                if role in (None, "assistant"):
                    d = max(0.0, t - start)
                    model_s += d
                    cur["model"] += d
                msg_open = None
        elif kind == "tool_execution_start":
            tool_open = t
        elif kind == "tool_execution_end":
            if tool_open is not None:
                d = max(0.0, t - tool_open)
                tool_s += d
                cur["tool"] += d
                tool_open = None

    if turn:
        per_turn.append((turn, cur["model"], cur["tool"]))
    # `span` is the agent's real lifetime: first event to agent_settled.
    # run_record.wall_seconds can be far larger, because a bash tool that leaves
    # a background process holding stdout keeps the pipe -- and therefore the
    # shell -- open long after the agent is done. svelte-2092 finished at 653s
    # and recorded 5103s. Span is the number to trust.
    return model_s, tool_s, turn, per_turn, span


def episodes(model_slug: str, task: str | None):
    base = RESULTS / model_slug
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not (d / "run_record.json").exists():
            continue
        if task and not (d.name == task or d.name.startswith(f"{task}__")):
            continue
        out.append(d)
    return out


def hms(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task")
    ap.add_argument("--per-turn", action="store_true")
    args = ap.parse_args()

    slug = args.model.replace("/", "_").replace(":", "_")
    dirs = episodes(slug, args.task)
    if not dirs:
        sys.exit(f"no runs under results/{slug}/")

    print("| Task | Result | Turns | Agent span | Model time | Tool time | Other | Model % | Wall (recorded) |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot_wall = tot_model = tot_tool = tot_span = 0.0
    for d in dirs:
        rec = json.loads((d / "run_record.json").read_text())
        split = split_time(d)
        if not split:
            continue
        model_s, tool_s, turns, per_turn, span = split
        wall = float(rec.get("wall_seconds") or 0)
        other = max(0.0, span - model_s - tool_s)
        stuck = wall - span
        vp = d / "verify.json"
        if vp.exists():
            v = json.loads(vp.read_text())
            res = "ERROR" if v.get("error") else ("PASS" if v.get("resolved") else "FAIL")
        else:
            res = "—"
        pct = (100 * model_s / span) if span else 0
        flag = f"{hms(wall)}" + (f"  (+{hms(stuck)} stuck)" if stuck > 30 else "")
        print(f"| `{d.name}` | {res} | {turns} | {hms(span)} | {hms(model_s)} | "
              f"{hms(tool_s)} | {hms(other)} | {pct:.0f}% | {flag} |")
        tot_wall += wall; tot_model += model_s; tot_tool += tool_s; tot_span += span

        if args.per_turn:
            print("\n  turn |  model |   tool")
            for t, m, tl in per_turn:
                print(f"  {t:>4} | {m:>5.1f}s | {tl:>5.1f}s")
            print()

    if tot_span:
        print(f"\n{len(dirs)} episodes · agent span {hms(tot_span)} · "
              f"model {hms(tot_model)} ({100*tot_model/tot_span:.0f}%) · "
              f"tool {hms(tot_tool)} ({100*tot_tool/tot_span:.0f}%) · "
              f"other {hms(tot_span - tot_model - tot_tool)}")
        print(f"recorded wall {hms(tot_wall)} -- {hms(tot_wall - tot_span)} of it "
              f"is post-agent pipe hang, not work")


if __name__ == "__main__":
    main()
