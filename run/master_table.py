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

# Splits of the same rows. The `resolved` column is dropped in these files --
# it is the split criterion, so repeating it in every row says nothing.
# ERROR and unverified runs appear in NEITHER: they have no verdict, and
# filing them under "unresolved" would count a harness failure as a model
# failure, which is the distinction this whole pipeline exists to preserve.
SPLITS = {
    "yes": (RESULTS / "resolved_table.md", RESULTS / "resolved_table.csv"),
    "no": (RESULTS / "unresolved_table.md", RESULTS / "unresolved_table.csv"),
}

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

        # Where the episode's time went. agent_span is the agent's real
        # lifetime; wall_seconds can be far larger when a bash tool leaves a
        # background process holding stdout after the agent is done.
        timing = metrics.get("timing") or {}

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
            "agent_span_s": timing.get("agent_span_s"),
            "model_time_s": timing.get("model_time_s"),
            "tool_time_s": timing.get("tool_time_s"),
            "model_time_pct": timing.get("model_time_pct"),
            "post_agent_hang_s": timing.get("post_agent_hang_s"),
            "cost_usd": cost,
            "llm_calls": calls or None,
            "patch_bytes": record.get("patch_bytes"),
            "f2p": f"{f2p.get('passed')}/{f2p.get('total')}" if f2p else "",
            "p2p": f"{p2p.get('passed')}/{p2p.get('total')}" if p2p else "",
            "timestamp_utc": record.get("timestamp_utc", ""),
        })
    # `model_attempt_n`: the Nth time THIS model has attempted THIS task. This
    # is what identifies an episode -- a task retried by the same model is a
    # genuinely different run, and overwriting it would erase the comparison.
    by_pair = {}
    for r in sorted(rows, key=lambda r: (r["model"], r["task_id"],
                                         r["timestamp_utc"] or "")):
        by_pair.setdefault((r["model"], r["task_id"]), []).append(r)
    for attempts in by_pair.values():
        for n, r in enumerate(attempts, 1):
            r["model_attempt_n"] = n
            r["model_attempts_total"] = len(attempts)

    # `pass_n`: the Nth time this task has been attempted, across all models,
    # ordered by run timestamp. Counted per task rather than per (task, model)
    # because the question it answers is "have we seen this task before" -- a
    # second attempt by a different model is still a second look at the task,
    # and its trajectory is informed by nothing the first run did.
    by_task = {}
    for r in sorted(rows, key=lambda r: (r["task_id"], r["timestamp_utc"] or "")):
        by_task.setdefault(r["task_id"], []).append(r)
    for attempts in by_task.values():
        for n, r in enumerate(attempts, 1):
            r["pass_n"] = n
            r["task_attempts_total"] = len(attempts)
            # How many *scored failures* preceded this attempt. Only "no"
            # counts: an ERROR or unverified earlier run tells us nothing about
            # difficulty, so counting it would inflate the apparent number of
            # tries a task needed.
            r["prior_failed_attempts"] = sum(
                1 for prev in attempts[: n - 1] if prev["resolved"] == "no"
            )
            # For a resolved row: the attempt on which it finally passed.
            # Blank on rows that did not pass -- there is nothing to report.
            r["passed_on_attempt"] = n if r["resolved"] == "yes" else None

        # Difficulty from observed outcomes, not from task features. A static
        # prior (gold patch size, F2P count) was tried and disagreed with
        # reality: it rated prettier-12930 "hard" (two models solved it) and
        # svelte-4332 "medium" (87 turns, failed, most expensive run in the
        # set). What a task actually costs an agent is not visible in its diff.
        #
        #   hard    no scored attempt resolved it
        #   easy    every scored attempt resolved it, none over 20 turns
        #   medium  resolved, but slowly -- or resolved by some models and not
        #           others, which is exactly the discriminating middle
        #   unrated no scored attempt yet (ERROR / not verified only)
        scored = [a for a in attempts if a["resolved"] in ("yes", "no")]
        won = [a for a in scored if a["resolved"] == "yes"]
        turns = [a["turns"] for a in won if a["turns"] is not None]
        if not scored:
            label = "unrated"
        elif not won:
            label = "hard"
        elif len(won) == len(scored) and (not turns or max(turns) <= 20):
            label = "easy"
        else:
            label = "medium"
        for r in attempts:
            r["difficulty"] = label

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
        "| # | Task ID | Task name | Lang | Diff | Model | Pass | Resolved | Turns | Wall | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        name = r["task_name"] or "*(task data not retained)*"
        if len(name) > 54:
            name = name[:54] + "…"
        out.append(
            f"| {i} | `{r['task_id']}` | {name} | {r['language'] or '?'} | "
            f"{r['difficulty']} | "
            f"{r['model'].split('/')[-1]} | {r['pass_n']}/{r['task_attempts_total']} | "
            f"{r['resolved']} | {r['turns'] or '—'} | "
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


FIELDS = ["task_id", "task_name", "language", "repo", "model", "provider",
          "difficulty", "resolved", "model_attempt_n", "model_attempts_total",
          "pass_n", "task_attempts_total", "passed_on_attempt",
          "prior_failed_attempts", "turns", "wall_seconds", "agent_span_s",
          "model_time_s", "tool_time_s", "model_time_pct", "post_agent_hang_s",
          "cost_usd", "llm_calls", "patch_bytes", "f2p", "p2p", "timestamp_utc"]


def write_csv(rows, path=None, fields=None):
    path = path or CSV_PATH
    fields = fields or FIELDS
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def attempt_cell(row, verdict):
    """How the attempt counter reads in each split.

    Resolved: "attempt 2 (after 1 fail)" -- the tries a task actually needed.
    Unresolved: "2 of 2" -- how many looks it has had so far.
    """
    if verdict == "yes":
        n, failed = row["passed_on_attempt"], row["prior_failed_attempts"]
        if not failed:
            return f"attempt {n}"
        return f"attempt {n} (after {failed} fail{'s' if failed > 1 else ''})"
    return f"attempt {row['pass_n']} of {row['task_attempts_total']}"


def write_split(rows, verdict, md_path, csv_path):
    """One verdict's rows, without the `resolved` column.

    Same schema as the master table minus the split criterion. Written even
    when empty, so a downstream reader gets a header rather than a missing file.
    """
    subset = [r for r in rows if r["resolved"] == verdict]
    fields = [f for f in FIELDS if f != "resolved"]
    write_csv(subset, csv_path, fields)

    label = "Resolved" if verdict == "yes" else "Unresolved"
    # Resolved rows answer "which try got there"; unresolved rows answer
    # "how many tries so far". Same underlying counter, different question.
    attempt_header = "Passed on" if verdict == "yes" else "Attempt"
    money = lambda v: f"${v:.4f}" if v is not None else "—"  # noqa: E731
    out = [
        f"# {label} tasks",
        "",
        f"Episodes the verifier scored **{label.lower()}**. Split out of "
        "`master_table.md`; regenerate with `python run/master_table.py`.",
        "",
        "Runs with no verdict (ERROR, not verified) appear in neither table — "
        "a harness failure is not a model failure.",
        "",
        f"| # | Task ID | Task name | Lang | Model | {attempt_header} | Turns | Wall | Cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(subset, 1):
        name = r["task_name"] or "*(task data not retained)*"
        if len(name) > 54:
            name = name[:54] + "…"
        out.append(
            f"| {i} | `{r['task_id']}` | {name} | {r['language'] or '?'} | "
            f"{r['model'].split('/')[-1]} | {attempt_cell(r, verdict)} | "
            f"{r['turns'] or '—'} | {hms(r['wall_seconds'])} | {money(r['cost_usd'])} |"
        )

    total = sum(r["cost_usd"] or 0 for r in subset)
    turns = [r["turns"] for r in subset if r["turns"] is not None]
    out += ["", f"**{len(subset)} episodes, ${total:.4f} total.**"]
    if turns:
        out.append(
            f" Turns: min {min(turns)}, median {sorted(turns)[len(turns)//2]}, "
            f"max {max(turns)}."
        )
    out.append("")
    md_path.write_text("\n".join(out))
    return subset


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

    counted = 0
    for verdict, (md_path, csv_path) in SPLITS.items():
        subset = write_split(rows, verdict, md_path, csv_path)
        counted += len(subset)
        print(f"  {md_path.name} / {csv_path.name}: {len(subset)} episodes, "
              f"${sum(r['cost_usd'] or 0 for r in subset):.4f}")

    # Anything with no verdict belongs to neither split, on purpose.
    no_verdict = len(rows) - counted
    if no_verdict:
        by_reason = {}
        for r in rows:
            if r["resolved"] not in SPLITS:
                by_reason[r["resolved"]] = by_reason.get(r["resolved"], 0) + 1
        print(f"  {no_verdict} episode(s) in neither split: {by_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
