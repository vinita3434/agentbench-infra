#!/usr/bin/env python3
"""Build results/agentbench.xlsx from the master table.

Tabs:
  Master              every episode, one row, with a stable episode id
  Common_All          tasks every model in the set has attempted
  <Frontier>_vs_<Open>  one tab per frontier x open-weight pair sharing tasks

Comparison tabs are generated, not enumerated, so adding a model adds its tabs
automatically. Only **frontier vs open-weight** pairs are produced: the question
this benchmark exists to answer is whether a self-hostable model can stand in
for a paid API, so opus-vs-sonnet (two frontier) and kimi-vs-deepseek (two open)
are noise. Add a model to MODEL_CLASS and its tabs appear on the next run.

    python run/build_workbook.py
"""
from __future__ import annotations

import csv
import pathlib
import sys
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results" / "master_table.csv"
XLSX_PATH = REPO_ROOT / "results" / "agentbench.xlsx"

# Which side of the comparison a model sits on. Keyed by the provider prefix of
# the model string, so a new model from a known provider is classified without
# an edit; unknown providers default to "open" and are flagged on stdout.
PROVIDER_CLASS = {
    "anthropic": "frontier",
    "openai": "frontier",
    "google": "frontier",
    "x-ai": "frontier",
    "moonshotai": "open",
    "qwen": "open",
    "deepseek": "open",
    "mistralai": "open",
    "meta-llama": "open",
    "zai-org": "open",
    "selfhosted": "open",     # anything served locally via SGLang
}


def model_class(model: str) -> str:
    provider = model.split("/")[0] if "/" in model else ""
    return PROVIDER_CLASS.get(provider, "open")


def short(model: str) -> str:
    return model.split("/")[-1]


def abbrev(name: str) -> str:
    """Shorten for a 31-char sheet title without mangling the model name.

    Only whole-token prefixes are dropped -- a naive substring replace turned
    "qwen3-coder-plus" into "qwen3r-plus".
    """
    for prefix in ("claude-", "moonshotai-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name[:14]


def hms(seconds) -> str:
    if not seconds:
        return ""
    m, s = divmod(int(float(seconds)), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def num(value, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def load():
    if not CSV_PATH.exists():
        sys.exit(f"no {CSV_PATH} -- run python run/master_table.py first")
    rows = list(csv.DictReader(CSV_PATH.open()))
    # Stable episode id: model slug + task + timestamp. Deterministic across
    # rebuilds, so a row keeps its id when new runs are appended.
    for r in rows:
        r["episode_id"] = (
            f"{short(r['model'])}|{r['task_id']}|{(r.get('timestamp_utc') or '')[:15]}"
        )
        r["class"] = model_class(r["model"])
    rows.sort(key=lambda r: (r["model"], r["task_id"], r.get("timestamp_utc") or ""))
    return rows


HEADER_FILL = "FF1F3864"
BAND_FILL = "FFF2F2F2"


def style_sheet(ws, header_rows=1, widths=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    for row in range(1, header_rows + 1):
        for cell in ws[row]:
            cell.font = Font(bold=True, color="FFFFFFFF")
            cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
    ws.freeze_panes = ws.cell(row=header_rows + 1, column=1)
    if ws.max_row > header_rows:
        ws.auto_filter.ref = (
            f"A{header_rows}:{get_column_letter(ws.max_column)}{ws.max_row}"
        )
    for i in range(1, ws.max_column + 1):
        letter = get_column_letter(i)
        longest = max(
            (len(str(ws.cell(row=r, column=i).value or ""))
             for r in range(1, ws.max_row + 1)), default=10)
        ws.column_dimensions[letter].width = min(
            max((widths or {}).get(i, 0), longest + 2, 9), 52)


def sheet_master(wb, rows):
    ws = wb.create_sheet("Master")
    ws.append(["Episode ID", "Task ID", "Task", "Model", "Class", "Difficulty",
               "Resolved", "Pass", "Turns", "Wall clock", "Wall (s)", "Cost (USD)",
               "Patch bytes", "F2P", "P2P", "Timestamp UTC"])
    for r in rows:
        ws.append([
            r["episode_id"], r["task_id"], r["task_name"], short(r["model"]),
            r["class"], r["difficulty"], r["resolved"],
            f"{r['pass_n']}/{r['task_attempts_total']}",
            num(r["turns"], int), hms(r["wall_seconds"]), num(r["wall_seconds"], int),
            num(r["cost_usd"]), num(r["patch_bytes"], int), r["f2p"], r["p2p"],
            r["timestamp_utc"],
        ])
    for row in ws.iter_rows(min_row=2, min_col=12, max_col=12):
        for cell in row:
            cell.number_format = '"$"#,##0.0000'
    style_sheet(ws)
    return ws


def _comparison_rows(rows, models):
    """Tasks every model in `models` has attempted, latest attempt each."""
    per = defaultdict(dict)
    for r in rows:
        m = short(r["model"])
        if m in models:
            per[r["task_id"]][m] = r      # rows are timestamp-sorted; last wins
    return {t: v for t, v in per.items() if len(v) == len(models)}


def sheet_comparison(wb, title, rows, models):
    shared = _comparison_rows(rows, models)
    if not shared:
        return None
    ws = wb.create_sheet(title[:31])

    top = ["", "", ""]
    sub = ["Task ID", "Task", "Difficulty"]
    for m in models:
        top += [m, "", "", ""]
        sub += ["Resolved", "Turns", "Wall clock", "Cost (USD)"]
    ws.append(top)
    ws.append(sub)
    for i, _ in enumerate(models):
        c = 4 + i * 4
        ws.merge_cells(start_row=1, start_column=c, end_row=1, end_column=c + 3)

    order = {"easy": 0, "medium": 1, "hard": 2, "unrated": 3}
    for task in sorted(shared, key=lambda t: (
            order.get(list(shared[t].values())[0]["difficulty"], 9), t)):
        v = shared[task]
        first = list(v.values())[0]
        line = [task, first["task_name"], first["difficulty"]]
        for m in models:
            r = v[m]
            line += [r["resolved"], num(r["turns"], int),
                     hms(r["wall_seconds"]), num(r["cost_usd"])]
        ws.append(line)

    # Totals: resolve rate over scored rows only, so ERROR / unverified never
    # count as failures.
    totals = ["TOTAL", f"{len(shared)} shared tasks", ""]
    for m in models:
        sub_rows = [v[m] for v in shared.values()]
        scored = [r for r in sub_rows if r["resolved"] in ("yes", "no")]
        won = sum(1 for r in scored if r["resolved"] == "yes")
        totals += [
            f"{won}/{len(scored)}",
            sum(num(r["turns"], int) or 0 for r in sub_rows),
            hms(sum(num(r["wall_seconds"], int) or 0 for r in sub_rows)),
            sum(num(r["cost_usd"]) or 0 for r in sub_rows),
        ]
    ws.append(totals)

    from openpyxl.styles import Font
    for cell in ws[ws.max_row]:
        cell.font = Font(bold=True)
    for i, _ in enumerate(models):
        col = 7 + i * 4
        for row in ws.iter_rows(min_row=3, min_col=col, max_col=col):
            for cell in row:
                cell.number_format = '"$"#,##0.0000'
    style_sheet(ws, header_rows=2)
    return ws


def main():
    try:
        from openpyxl import Workbook
    except ImportError:
        sys.exit("pip install openpyxl")

    rows = load()
    models = sorted({short(r["model"]) for r in rows})
    cls = {short(r["model"]): r["class"] for r in rows}
    frontier = [m for m in models if cls[m] == "frontier"]
    openw = [m for m in models if cls[m] == "open"]

    wb = Workbook()
    wb.remove(wb.active)
    sheet_master(wb, rows)
    print(f"Master: {len(rows)} episodes, {len(models)} models")

    # Tasks a whole group of models has attempted -- the fully controlled
    # comparison. Try all models first; if nothing is shared by all of them,
    # fall back to the largest subset that does share tasks, so the tab is
    # useful rather than empty.
    from itertools import combinations
    best = None
    for size in range(len(models), 2, -1):
        for combo in combinations(models, size):
            shared = _comparison_rows(rows, list(combo))
            if shared and (best is None or len(shared) > len(best[1])):
                best = (list(combo), shared)
        if best:
            break
    if best:
        ws = sheet_comparison(wb, "Common", rows, best[0])
        print(f"Common: {len(best[1])} tasks shared by {', '.join(best[0])}")
    else:
        print("Common: no task attempted by 3+ models yet")

    # Frontier x open only. A new open model pairs against every frontier model
    # automatically; frontier-vs-frontier and open-vs-open are never emitted.
    for f in frontier:
        for o in openw:
            title = f"{abbrev(f)}_vs_{abbrev(o)}"
            ws = sheet_comparison(wb, title, rows, [f, o])
            if ws:
                print(f"{ws.title}: {ws.max_row - 3} shared tasks")

    XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX_PATH)
    print(f"\n-> {XLSX_PATH}")
    print(f"   tabs: {', '.join(wb.sheetnames)}")


if __name__ == "__main__":
    main()
