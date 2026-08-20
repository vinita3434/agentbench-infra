#!/usr/bin/env python3
"""Turn a captured Pi run into trajectory metrics.

Reads results/<model>/<task>/events.timed.jsonl (produced by _stamp_stream.py;
each line is {"_t": seconds, "event": <pi event>}) and derives:

  submitted          did the run produce a non-empty candidate patch
  turns              number of agent turns
  turns_to_submit    turns until the patch existed (== turns here; the agent
                     submits by editing files, captured at run end)
  per_turn[]         for each turn: ttft, duration, reasoning snippet, actions
  ttft_trajectory    per-turn time-to-first-token (turn_start -> first delta)
  tokens             input/output if Pi surfaced usage (may be absent on OpenRouter)
  wall_seconds       from run_record.json

Pi event shapes used (docs/json.md): turn_start / turn_end, message_update with
assistantMessageEvent {type: text_delta|thinking_delta, ...},
tool_execution_start {toolName, args}. Parsing is defensive: unknown shapes are
skipped, not fatal.

Usage:
  extract_metrics.py <task_dir>     # write <task_dir>/metrics.json, print summary
  extract_metrics.py --all          # every results/<model>/<task>, print table
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

IN_KEYS = ("input_tokens", "prompt_tokens", "inputTokens", "promptTokens")
OUT_KEYS = ("output_tokens", "completion_tokens", "outputTokens", "completionTokens")
_ARG_HINT_KEYS = ("path", "file_path", "filePath", "command", "cmd", "pattern",
                  "query", "url", "old_string", "content")


def _first_int(d, keys):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return int(d[k])
    return 0


def _scan_usage(obj, acc):
    if isinstance(obj, dict):
        if any(k in obj for k in IN_KEYS) or any(k in obj for k in OUT_KEYS):
            acc["in"] += _first_int(obj, IN_KEYS)
            acc["out"] += _first_int(obj, OUT_KEYS)
        for v in obj.values():
            _scan_usage(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _scan_usage(v, acc)


def _delta_text(ame):
    """Pull the streamed text out of an assistantMessageEvent, whatever it's called."""
    for k in ("text", "delta", "thinking", "content", "value"):
        v = ame.get(k)
        if isinstance(v, str):
            return v
    return ""


def _summarize_args(args):
    if isinstance(args, dict):
        for k in _ARG_HINT_KEYS:
            if isinstance(args.get(k), str) and args[k]:
                s = args[k].replace("\n", " ")
                return s[:80] + ("..." if len(s) > 80 else "")
        # fall back to compact json
        s = json.dumps(args)[:80]
        return s
    if args is None:
        return ""
    return str(args)[:80]


def load_timed_events(task_dir):
    """Prefer events.timed.jsonl; fall back to pi_log.jsonl (no timings)."""
    timed = task_dir / "events.timed.jsonl"
    if timed.exists() and timed.stat().st_size > 0:
        out = []
        for line in timed.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out.append((rec.get("_t"), rec.get("event", {})))
            except json.JSONDecodeError:
                continue
        return out, True
    raw = task_dir / "pi_log.jsonl"
    out = []
    if raw.exists():
        for line in raw.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append((None, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return out, False


def time_split(events):
    """Split the episode into model time, tool time, and everything else.

    An agentic episode alternates between waiting on the model and running
    commands locally, and the two are governed by completely different things:
    model time scales with context and is what a serving stack controls, tool
    time is npm installs and test suites and is unaffected by the model.

    `agent_span_s` is first event to last -- the agent's real lifetime. It is
    NOT the same as run_record.wall_seconds: a bash tool that leaves a
    background process holding stdout keeps the pipe, and therefore the shell,
    open long after the agent finished. One svelte episode finished in 653s and
    recorded 5103s. Span is the number to reason with; the difference is
    reported as `post_agent_hang_s` rather than hidden.
    """
    span = model_s = tool_s = 0.0
    msg_open = None
    tool_open = None

    for offset, event in events:
        if offset is None:
            continue
        span = max(span, offset)
        kind = event.get("type")
        if kind == "message_start":
            msg_open = (offset, (event.get("message") or {}).get("role"))
        elif kind == "message_end":
            if msg_open:
                start, role = msg_open
                # Only assistant messages cost model time; the user turn is the
                # harness handing over a prompt it already has.
                if role in (None, "assistant"):
                    model_s += max(0.0, offset - start)
                msg_open = None
        elif kind == "tool_execution_start":
            tool_open = offset
        elif kind == "tool_execution_end":
            if tool_open is not None:
                tool_s += max(0.0, offset - tool_open)
                tool_open = None

    return {
        "agent_span_s": round(span, 2),
        "model_time_s": round(model_s, 2),
        "tool_time_s": round(tool_s, 2),
        # Inter-event gaps: Pi's own processing, patch handling, stream I/O.
        "other_time_s": round(max(0.0, span - model_s - tool_s), 2),
        "model_time_pct": round(100 * model_s / span, 1) if span else None,
        "tool_time_pct": round(100 * tool_s / span, 1) if span else None,
    }


def metrics_for(task_dir):
    task_dir = pathlib.Path(task_dir)
    events, have_timing = load_timed_events(task_dir)

    turns = []
    cur = None
    tokens = {"in": 0, "out": 0}

    def open_turn(t):
        return {"index": len(turns) + 1, "start_t": t, "first_token_t": None,
                "end_t": None, "reasoning": "", "answer": "", "actions": []}

    for t, ev in events:
        etype = str(ev.get("type", ""))
        _scan_usage(ev, tokens)

        if etype == "turn_start":
            if cur is not None:            # defensive: close an unclosed turn
                turns.append(cur)
            cur = open_turn(t)
        elif etype == "message_update":
            ame = ev.get("assistantMessageEvent") or {}
            atype = str(ame.get("type", ""))
            txt = _delta_text(ame)
            if cur is None:
                cur = open_turn(t)
            if txt and cur["first_token_t"] is None:
                cur["first_token_t"] = t
            if atype == "thinking_delta":
                cur["reasoning"] += txt
            elif atype == "text_delta":
                cur["answer"] += txt
        elif etype == "tool_execution_start":
            if cur is None:
                cur = open_turn(t)
            cur["actions"].append({
                "tool": ev.get("toolName", ""),
                "args": _summarize_args(ev.get("args")),
            })
        elif etype == "turn_end":
            if cur is None:
                cur = open_turn(t)
            cur["end_t"] = t
            turns.append(cur)
            cur = None
    if cur is not None:
        turns.append(cur)

    # Assemble per-turn view + TTFT trajectory.
    per_turn = []
    ttft_traj = []
    for tn in turns:
        ttft = None
        if have_timing and tn["start_t"] is not None and tn["first_token_t"] is not None:
            ttft = round(tn["first_token_t"] - tn["start_t"], 3)
        dur = None
        if have_timing and tn["start_t"] is not None and tn["end_t"] is not None:
            dur = round(tn["end_t"] - tn["start_t"], 3)
        reasoning = tn["reasoning"].strip().replace("\n", " ")
        per_turn.append({
            "turn": tn["index"],
            "ttft_s": ttft,
            "duration_s": dur,
            "reasoning": (reasoning[:400] + "...") if len(reasoning) > 400 else reasoning,
            "actions": tn["actions"],
        })
        if ttft is not None:
            ttft_traj.append(ttft)

    # submitted / wall from the run record + patch file.
    run_record = {}
    rr = task_dir / "run_record.json"
    if rr.exists():
        try:
            run_record = json.loads(rr.read_text())
        except json.JSONDecodeError:
            pass
    patch = task_dir / "candidate.patch"
    submitted = patch.exists() and patch.stat().st_size > 0

    metrics = {
        "task_id": run_record.get("task_id", task_dir.name),
        "model": run_record.get("model"),
        "submitted": submitted,
        "turns": len(turns),
        "turns_to_submit": len(turns) if submitted else None,
        "ttft_trajectory": ttft_traj,
        "ttft_available": have_timing,
        "tokens": {"input": tokens["in"], "output": tokens["out"],
                   "available": (tokens["in"] + tokens["out"]) > 0},
        "wall_seconds": run_record.get("wall_seconds"),
        "pi_exit_code": run_record.get("pi_exit_code"),
        "per_turn": per_turn,
    }

    # Where the episode's time actually went. Captured here, at extraction
    # time, so every run carries it -- rather than being reconstructed later
    # from the raw stream and argued about.
    metrics["timing"] = time_split(events)
    wall = run_record.get("wall_seconds")
    span = metrics["timing"]["agent_span_s"]
    if wall is not None and span:
        # Positive values mean the shell kept waiting after the agent was done,
        # usually a background process left holding stdout by a bash tool.
        metrics["timing"]["post_agent_hang_s"] = round(max(0.0, wall - span), 2)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", nargs="?", help="a results/<model>/<task> directory")
    ap.add_argument("--all", action="store_true", help="process every results task dir")
    args = ap.parse_args()

    dirs = []
    if args.all:
        for model_dir in sorted(RESULTS.glob("*")):
            if not model_dir.is_dir() or model_dir.name in ("serve",):
                continue
            for td in sorted(model_dir.iterdir()):
                if td.is_dir() and (td / "run_record.json").exists():
                    dirs.append(td)
    elif args.task_dir:
        dirs = [pathlib.Path(args.task_dir)]
    else:
        ap.error("give a task_dir or --all")

    for td in dirs:
        m = metrics_for(td)
        (td / "metrics.json").write_text(json.dumps(m, indent=2))
        ttft = m["ttft_trajectory"]
        ttft_str = f"{ttft[0]}s..{ttft[-1]}s" if ttft else "n/a"
        print(f"{m['task_id']}: submitted={m['submitted']} turns={m['turns']} "
              f"ttft(first..last)={ttft_str} "
              f"tokens={m['tokens']['input']}/{m['tokens']['output']} "
              f"wall={m['wall_seconds']}s -> {td/'metrics.json'}")


if __name__ == "__main__":
    main()
