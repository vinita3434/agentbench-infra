#!/usr/bin/env python3
"""Master table: every OpenRouter episode, one row each, with cost.

Writes results/master_table.md (readable) and results/master_table.csv
(machine-readable). Regenerate it; never hand-edit it.

    python run/master_table.py

**On cost.** Pi's event stream carries a `usage` object on every assistant
`message_end`, and that object describes ONE call: its own input/output tokens
and its own dollar cost. Episode cost is therefore the SUM across calls.

`totalTokens` on the same object is not summable -- it grows monotonically
because each turn resends the whole conversation, so adding it up counts the
prompt again every turn. Cost is per-call; token totals are cumulative context.
Mixing them is the same class of error as dividing a server-wide counter by a
per-request denominator.

Absent stays absent: a run whose log carries no usage (the agent never reached
the model) shows cost as None, not $0.00 -- those are different facts.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "run"))

from task_artifact import task_summary  # noqa: E402

RESULTS = REPO_ROOT / "results"
MD_PATH = RESULTS / "master_table.md"
CSV_PATH = RESULTS / "master_table.csv"

# Language for tasks whose data was dropped when the 50-task set was
# regenerated. Derived from the repo, which the run record always retains.
REPO_LANG = {
    "langchain-ai/langchain": "Python",
    "keras-team/keras": "Python",
    "prettier/prettier": "JavaScript",
    "serverless/serverless": "JavaScript",
    "sveltejs/svelte": "JavaScript",
    "mui/material-ui": "TypeScript",
    "yt-dlp/yt-dlp": "Python",
    "apache/dubbo": "Java",
}


def episode_cost(log_path: pathlib.Path):
    """(total_usd, n_calls) summed over assistant message_end events.

    Returns (None, 0) when the log carries no usage at all -- a run that never
    reached the model is not a run that cost nothing to think about.
    """
    if not log_path.exists():
        return None, 0
    total, calls = 0.0, 0
    with log_path.open() as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message_end":
                continue
            usage = (event.get("message") or {}).get("usage") or {}
            cost = (usage.get("cost") or {}).get("total")
            if cost is not None:
                total += cost
                calls += 1
    return (total, calls) if calls else (None, 0)


def load_rows():
    rows = []
    for record_path in sorted(RESULTS.glob("*/*/run_record.json")):
        directory = record_path.parent
        # results/_gold holds verifier self-checks, not agent runs.
        if directory.parent.name in {"_gold", "serve", "artifacts"}:
            continue
        record = json.loads(record_path.read_text())
        task = record.get("task_id", directory.name)

        task_file = REPO_ROOT / "tasks" / "data" / f"{task}.json"
        meta = json.loads(task_file.read_text()) if task_file.exists() else {}
        name = task_summary(meta)[0] if meta else ""

        metrics_path = directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

        verify_path = directory / "verify.json"
        verify = json.loads(verify_path.read_text()) if verify_path.exists() else None
        if verify is None:
            resolved = "not verified"
        elif verify.get("error"):
            # Harness could not score it -- never conflated with a model failure.
            resolved = "ERROR"
        else:
            resolved = "yes" if verify.get("resolved") else "no"

        cost, calls = episode_cost(directory / "pi_log.jsonl")
        f2p = (verify or {}).get("f2p") or {}
        p2p = (verify or {}).get("p2p") or {}

        rows.append({
            "task_id": task,
            "task_name": name,
            "language": meta.get("language") or REPO_LANG.get(record.get("repo", ""), ""),
            "repo": record.get("repo", ""),
            "model": record.get("model", ""),
            "provider": record.get("provider", ""),
            "resolved": resolved,
            "turns": metrics.get("turns"),
            "wall_seconds": record.get("wall_seconds"),
            "cost_usd": cost,
            "llm_calls": calls or None,
            "patch_bytes": record.get("patch_bytes"),
            "f2p": f"{f2p.get('passed')}/{f2p.get('total')}" if f2p else "",
            "p2p": f"{p2p.get('passed')}/{p2p.get('total')}" if p2p else "",
            "timestamp_utc": record.get("timestamp_utc", ""),
        })
    rows.sort(key=lambda r: (r["model"], -(r["cost_usd"] or 0)))
    return rows


def hms(seconds):
    if seconds is None:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def write_markdown(rows):
    money = lambda v: f"${v:.4f}" if v is not None else "—"  # noqa: E731
    out = [
        "# Master table — OpenRouter episodes",
        "",
        "Every agent run against a frontier model via OpenRouter, one row each.",
        "Regenerated by `python run/master_table.py` — do not hand-edit.",
        "",
        "Cost is the **sum of per-call costs** from Pi's event stream. Each",
        "assistant message carries its own call cost; `totalTokens` on the same",
        "object grows monotonically (context accumulating) and is not summable.",
        "",
        "| # | Task ID | Task name | Lang | Model | Resolved | Turns | Wall | Cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        name = r["task_name"] or "*(task data not retained)*"
        if len(name) > 54:
            name = name[:54] + "…"
        out.append(
            f"| {i} | `{r['task_id']}` | {name} | {r['language'] or '?'} | "
            f"{r['model'].split('/')[-1]} | {r['resolved']} | {r['turns'] or '—'} | "
            f"{hms(r['wall_seconds'])} | {money(r['cost_usd'])} |"
        )

    out += ["", "## Per model", "",
            "| Model | Runs | Resolved | Total cost | Avg per resolved |",
            "|---|---|---|---|---|"]
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        # Denominator is scored runs only: ERROR and unverified rows have no
        # defensible place in a resolve rate.
        scored = [r for r in sub if r["resolved"] in ("yes", "no")]
        won = [r for r in sub if r["resolved"] == "yes"]
        total = sum(r["cost_usd"] or 0 for r in sub)
        avg = (sum(r["cost_usd"] or 0 for r in won) / len(won)) if won else None
        out.append(
            f"| {model} | {len(sub)} | {len(won)}/{len(scored)} scored | "
            f"${total:.4f} | {money(avg)} |"
        )

    grand = sum(r["cost_usd"] or 0 for r in rows)
    out += ["", f"**{len(rows)} episodes, ${grand:.4f} total.**", ""]
    MD_PATH.write_text("\n".join(out))


def write_csv(rows):
    fields = ["task_id", "task_name", "language", "repo", "model", "provider",
              "resolved", "turns", "wall_seconds", "cost_usd", "llm_calls",
              "patch_bytes", "f2p", "p2p", "timestamp_utc"]
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_rows()
    if not rows:
        print("no runs found under results/", file=sys.stderr)
        return 1
    write_markdown(rows)
    write_csv(rows)
    total = sum(r["cost_usd"] or 0 for r in rows)
    print(f"{len(rows)} episodes, ${total:.4f} total")
    print(f"  {MD_PATH}")
    print(f"  {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
