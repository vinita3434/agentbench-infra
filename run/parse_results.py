#!/usr/bin/env python3
"""Summarize Pi JSON logs into a per-task table.

Reads results/<model>/<task_id>/ directories and, for each, extracts:
    turns        number of assistant turns
    tool_calls   number of tool calls the agent made
    in_tokens    total prompt/input tokens across turns
    out_tokens   total completion/output tokens across turns
    patch        whether a non-empty candidate patch was produced
    wall_s       wall-clock seconds (from run_record.json)

Pi emits `--mode json` as JSON lines (one event per line). The exact event
schema isn't pinned across Pi versions, so this parser is deliberately
defensive: it scans each event for known token/turn/tool-call shapes rather
than assuming one layout. Unknown lines are ignored.

Usage:
    python run/parse_results.py                 # all models under results/
    python run/parse_results.py --model anthropic__claude-sonnet-4-6
    python run/parse_results.py --json          # machine-readable output
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# Token-count key aliases seen across OpenAI-compatible / Pi usage payloads.
IN_KEYS = ("input_tokens", "prompt_tokens", "inputTokens", "promptTokens")
OUT_KEYS = ("output_tokens", "completion_tokens", "outputTokens", "completionTokens")


def _first_int(d, keys):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return int(d[k])
    return 0


def _scan_usage(obj, acc):
    """Recursively find usage-like dicts and accumulate in/out tokens."""
    if isinstance(obj, dict):
        if any(k in obj for k in IN_KEYS) or any(k in obj for k in OUT_KEYS):
            acc["in"] += _first_int(obj, IN_KEYS)
            acc["out"] += _first_int(obj, OUT_KEYS)
        for v in obj.values():
            _scan_usage(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _scan_usage(v, acc)


def parse_log(log_path):
    turns = 0
    tool_calls = 0
    tokens = {"in": 0, "out": 0}
    if not log_path.exists():
        return {"turns": 0, "tool_calls": 0, "in_tokens": 0, "out_tokens": 0,
                "parsed": False}

    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = str(event.get("type", "")).lower()

        # Count assistant turns: an assistant message / turn-complete event.
        role = event.get("role") or event.get("message", {}).get("role") \
            if isinstance(event.get("message"), dict) else event.get("role")
        if role == "assistant" or etype in ("assistant", "turn", "turn_complete",
                                            "message", "response"):
            turns += 1

        # Count tool calls wherever they appear.
        if etype in ("tool_call", "tool_use", "toolcall"):
            tool_calls += 1
        for container in (event, event.get("message") if isinstance(event.get("message"), dict) else None):
            if isinstance(container, dict):
                tcs = container.get("tool_calls") or container.get("toolCalls")
                if isinstance(tcs, list):
                    tool_calls += len(tcs)

        _scan_usage(event, tokens)

    return {"turns": turns, "tool_calls": tool_calls,
            "in_tokens": tokens["in"], "out_tokens": tokens["out"],
            "parsed": True}


def load_run_record(task_dir):
    rec = task_dir / "run_record.json"
    if rec.exists():
        try:
            return json.loads(rec.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def patch_produced(task_dir):
    p = task_dir / "candidate.patch"
    return p.exists() and p.stat().st_size > 0


def collect(model_filter=None):
    rows = []
    if not RESULTS.exists():
        return rows
    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "serve":
            continue
        if model_filter and model_dir.name != model_filter:
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            stats = parse_log(task_dir / "pi_log.jsonl")
            record = load_run_record(task_dir)
            rows.append({
                "model": model_dir.name,
                "task": task_dir.name,
                "turns": stats["turns"],
                "tool_calls": stats["tool_calls"],
                "in_tokens": stats["in_tokens"],
                "out_tokens": stats["out_tokens"],
                "patch": patch_produced(task_dir),
                "wall_s": record.get("wall_seconds", ""),
                "pi_rc": record.get("pi_exit_code", ""),
            })
    return rows


def print_table(rows):
    if not rows:
        print("No results found under results/. Run run/run_task.sh first.")
        return
    headers = ["model", "task", "turns", "tools", "in_tok", "out_tok",
               "patch", "wall_s", "rc"]
    keys = ["model", "task", "turns", "tool_calls", "in_tokens", "out_tokens",
            "patch", "wall_s", "pi_rc"]
    table = [headers]
    for r in rows:
        table.append([
            r["model"], r["task"], str(r["turns"]), str(r["tool_calls"]),
            str(r["in_tokens"]), str(r["out_tokens"]),
            "yes" if r["patch"] else "NO", str(r["wall_s"]), str(r["pi_rc"]),
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    for ri, row in enumerate(table):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if ri == 0:
            print("  ".join("-" * widths[i] for i in range(len(headers))))

    n = len(rows)
    with_patch = sum(1 for r in rows if r["patch"])
    print()
    print(f"tasks: {n}   with-patch: {with_patch}   "
          f"patch-rate: {with_patch / n:.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="restrict to one results/<model> dir")
    ap.add_argument("--json", action="store_true", help="emit JSON not a table")
    args = ap.parse_args()

    rows = collect(args.model)
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
