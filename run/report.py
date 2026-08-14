#!/usr/bin/env python3
"""Consolidate per-task metrics + verification into one view.

Merges, for every results/<model>/<task>:
  - trajectory metrics  (run/extract_metrics.py: submitted, turns, turns_to_submit,
                         per-turn reasoning/action, TTFT trajectory, tokens)
  - verification        (run/verify_task.sh -> verify.json: F2P%, P2P%, resolved)

Outputs:
  - a terminal table
  - results/summary.json   (machine-readable rollup + per-task rows)
  - results/report.html    (browsable: summary + per-turn cards + TTFT chart)

Usage:
  python run/report.py           # all models under results/
  python run/report.py --open    # also print the file:// URL to open
"""
import argparse
import html
import json
import pathlib
import statistics

import extract_metrics  # sibling module

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def median(xs):
    return round(statistics.median(xs), 3) if xs else None


def load_verify(task_dir):
    v = task_dir / "verify.json"
    if v.exists():
        try:
            return json.loads(v.read_text())
        except json.JSONDecodeError:
            return None
    return None


def collect():
    rows = []
    if not RESULTS.exists():
        return rows
    for model_dir in sorted(RESULTS.glob("*")):
        if not model_dir.is_dir() or model_dir.name in ("serve", "_gold"):
            continue
        for td in sorted(model_dir.iterdir()):
            if not td.is_dir() or not (td / "run_record.json").exists():
                continue
            m = extract_metrics.metrics_for(td)
            (td / "metrics.json").write_text(json.dumps(m, indent=2))
            v = load_verify(td)
            rows.append({"model": model_dir.name, "task": td.name,
                         "metrics": m, "verify": v})
    return rows


# --------------------------------------------------------------------------- #
# terminal table
# --------------------------------------------------------------------------- #
def print_table(rows):
    if not rows:
        print("No completed runs under results/. Run run/run_task.sh first.")
        return
    hdr = ["model", "task", "sub", "resolved", "F2P", "P2P", "turns", "ttft_p50", "wall_s"]
    table = [hdr]
    for r in rows:
        m, v = r["metrics"], r["verify"]
        f2p = f'{v["f2p"]["passed"]}/{v["f2p"]["total"]}' if v else "-"
        p2p = f'{v["p2p"]["passed"]}/{v["p2p"]["total"]}' if v else "-"
        resolved = ("yes" if v["resolved"] else "no") if v else "unverified"
        table.append([
            r["model"][:26], r["task"][:28],
            "yes" if m["submitted"] else "NO",
            resolved, f2p, p2p, str(m["turns"]),
            str(median(m["ttft_trajectory"]) if m["ttft_trajectory"] else "-"),
            str(m["wall_seconds"]),
        ])
    w = [max(len(row[i]) for row in table) for i in range(len(hdr))]
    for ri, row in enumerate(table):
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(row)))
        if ri == 0:
            print("  ".join("-" * w[i] for i in range(len(hdr))))

    n = len(rows)
    sub = sum(1 for r in rows if r["metrics"]["submitted"])
    ver = [r for r in rows if r["verify"]]
    res = sum(1 for r in ver if r["verify"]["resolved"])
    print()
    print(f"tasks: {n}   submitted: {sub}   verified: {len(ver)}   resolved: {res}"
          + (f"   resolve-rate: {res/len(ver):.0%}" if ver else ""))


# --------------------------------------------------------------------------- #
# summary.json rollup
# --------------------------------------------------------------------------- #
def build_summary(rows):
    ver = [r for r in rows if r["verify"]]
    all_ttft = [t for r in rows for t in r["metrics"]["ttft_trajectory"]]
    return {
        "totals": {
            "tasks": len(rows),
            "submitted": sum(1 for r in rows if r["metrics"]["submitted"]),
            "verified": len(ver),
            "resolved": sum(1 for r in ver if r["verify"]["resolved"]),
            "resolve_rate": (round(sum(1 for r in ver if r["verify"]["resolved"]) / len(ver), 3)
                             if ver else None),
            "median_ttft_s": median(all_ttft),
            "median_turns": median([r["metrics"]["turns"] for r in rows]),
        },
        "tasks": [{
            "model": r["model"], "task": r["task"],
            "submitted": r["metrics"]["submitted"],
            "turns": r["metrics"]["turns"],
            "turns_to_submit": r["metrics"]["turns_to_submit"],
            "ttft_trajectory": r["metrics"]["ttft_trajectory"],
            "tokens": r["metrics"]["tokens"],
            "wall_seconds": r["metrics"]["wall_seconds"],
            "resolved": r["verify"]["resolved"] if r["verify"] else None,
            "f2p": r["verify"]["f2p"] if r["verify"] else None,
            "p2p": r["verify"]["p2p"] if r["verify"] else None,
        } for r in rows],
    }


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def ttft_svg(ttft, w=360, h=90, pad=18):
    if not ttft:
        return '<span class="muted">no TTFT (native/no-timing run)</span>'
    if len(ttft) == 1:
        return f'<span class="chip">TTFT {ttft[0]}s (1 turn)</span>'
    lo, hi = min(ttft), max(ttft)
    span = (hi - lo) or 1.0
    n = len(ttft)
    pts = []
    for i, v in enumerate(ttft):
        x = pad + (w - 2 * pad) * i / (n - 1)
        y = h - pad - (h - 2 * pad) * (v - lo) / span
        pts.append(f"{x:.1f},{y:.1f}")
    dots = "".join(f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="2.5"/>'
                   for p in pts)
    return (f'<svg viewBox="0 0 {w} {h}" class="spark" role="img" '
            f'aria-label="TTFT per turn">'
            f'<polyline fill="none" stroke-width="1.5" points="{" ".join(pts)}"/>'
            f'{dots}<text x="{pad}" y="12" class="axis">{hi}s</text>'
            f'<text x="{pad}" y="{h-4}" class="axis">{lo}s</text></svg>')


def esc(s):
    return html.escape(str(s if s is not None else ""))


def task_html(r):
    m, v = r["metrics"], r["verify"]
    if v:
        badge = ('<span class="badge ok">RESOLVED</span>' if v["resolved"]
                 else '<span class="badge bad">not resolved</span>')
        f2p = v["f2p"]; p2p = v["p2p"]
        tests = ""
        for label, grp in (("F2P", f2p), ("P2P", p2p)):
            items = "".join(
                f'<li class="{"tpass" if t["passed"] else "tfail"}">'
                f'{"✓" if t["passed"] else "✗"} {esc(t["test"].split("->")[-1])}</li>'
                for t in grp["tests"])
            tests += (f'<div class="testgrp"><b>{label} {grp["passed"]}/{grp["total"]} '
                      f'({grp["pct"]}%)</b><ul>{items}</ul></div>')
    else:
        badge = '<span class="badge unk">unverified</span>'
        tests = '<div class="muted">no verify.json (run run/verify_task.sh)</div>'

    turns = ""
    for pt in m["per_turn"]:
        acts = "".join(f'<span class="chip">{esc(a["tool"])}'
                       + (f' <span class="muted">{esc(a["args"])}</span>' if a["args"] else "")
                       + '</span>' for a in pt["actions"])
        ttft = f'{pt["ttft_s"]}s' if pt["ttft_s"] is not None else "–"
        dur = f'{pt["duration_s"]}s' if pt["duration_s"] is not None else "–"
        reasoning = esc(pt["reasoning"]) or '<span class="muted">(no reasoning captured)</span>'
        turns += (f'<div class="turn"><div class="turnhead">'
                  f'<b>turn {pt["turn"]}</b>'
                  f'<span class="t">TTFT {ttft}</span><span class="t">dur {dur}</span></div>'
                  f'<div class="reason">{reasoning}</div>'
                  f'<div class="acts">{acts or "<span class=muted>no tool calls</span>"}</div></div>')

    tok = m["tokens"]
    tokstr = (f'{tok["input"]}/{tok["output"]}' if tok["available"] else "n/a")
    return f"""
    <section class="card">
      <h3>{esc(r["task"])} {badge}</h3>
      <div class="meta">
        <span>model <b>{esc(m["model"])}</b></span>
        <span>submitted <b>{"yes" if m["submitted"] else "NO"}</b></span>
        <span>turns <b>{m["turns"]}</b></span>
        <span>turns→submit <b>{esc(m["turns_to_submit"])}</b></span>
        <span>tokens(in/out) <b>{tokstr}</b></span>
        <span>wall <b>{esc(m["wall_seconds"])}s</b></span>
      </div>
      <div class="cols">
        <div class="left"><div class="lbl">TTFT trajectory</div>{ttft_svg(m["ttft_trajectory"])}{tests}</div>
        <div class="right"><div class="lbl">trajectory ({m["turns"]} turns)</div>{turns or '<div class="muted">no turns captured</div>'}</div>
      </div>
    </section>"""


CSS = """
  :root { color-scheme: light dark; --bg:#fff; --fg:#191a22; --mut:#6a6c7a;
          --line:#e6e6ef; --card:#f8f8fc; --ok:#16a34a; --bad:#dc2626; --unk:#d97706;
          --chip:#eef2ff; --accent:#4f46e5; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1115; --fg:#e6e6e6; --mut:#9aa0aa; --line:#242832; --card:#161922;
            --chip:#1e2233; --accent:#818cf8; } }
  :root[data-theme="dark"] { --bg:#0f1115; --fg:#e6e6e6; --mut:#9aa0aa; --line:#242832;
            --card:#161922; --chip:#1e2233; --accent:#818cf8; }
  :root[data-theme="light"] { --bg:#fff; --fg:#1a1a1a; --mut:#6b7280; --line:#e5e7eb;
            --card:#fafafa; --chip:#eef2ff; --accent:#4f46e5; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--fg); padding:24px; max-width:1100px; margin:auto; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--mut); margin-bottom:20px; }
  .kpis { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:24px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:12px 16px; min-width:120px; }
  .kpi .n { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }
  .kpi .l { color:var(--mut); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:16px 18px; margin-bottom:18px; }
  .card h3 { margin:0 0 10px; font-size:15px; display:flex; align-items:center; gap:10px; }
  .meta { display:flex; flex-wrap:wrap; gap:14px; color:var(--mut); font-size:13px; margin-bottom:12px; }
  .meta b { color:var(--fg); }
  .cols { display:grid; grid-template-columns: 380px 1fr; gap:18px; }
  @media (max-width:820px) { .cols { grid-template-columns:1fr; } }
  .lbl { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); margin-bottom:6px; }
  .spark { width:100%; max-width:380px; border:1px solid var(--line); border-radius:8px; background:var(--bg); }
  .spark polyline, .spark circle { stroke:var(--accent); fill:var(--accent); }
  .spark polyline { fill:none; } .spark circle { stroke:none; }
  .axis { fill:var(--mut); font-size:9px; }
  .right { min-width:0; }
  .turn { border-left:2px solid var(--line); padding:6px 0 6px 12px; margin-bottom:8px; }
  .turnhead { display:flex; gap:12px; align-items:baseline; }
  .turnhead .t { color:var(--mut); font-size:12px; }
  .reason { margin:4px 0; white-space:pre-wrap; overflow-wrap:anywhere; }
  .acts { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { background:var(--chip); border-radius:6px; padding:1px 8px; font-size:12px;
          overflow-wrap:anywhere; }
  .muted { color:var(--mut); }
  .badge { font-size:11px; padding:2px 8px; border-radius:20px; font-weight:600; }
  .badge.ok { background:var(--ok); color:#fff; } .badge.bad { background:var(--bad); color:#fff; }
  .badge.unk { background:var(--unk); color:#fff; }
  .testgrp { margin-top:10px; } .testgrp ul { margin:4px 0; padding-left:18px; }
  .tpass { color:var(--ok); } .tfail { color:var(--bad); }
"""


def render_body(rows, summary):
    t = summary["totals"]
    cards = "".join(task_html(r) for r in rows) or '<p class="muted">No runs yet.</p>'
    rr = f'{t["resolve_rate"]:.0%}' if t["resolve_rate"] is not None else "–"
    mttft = t["median_ttft_s"] if t["median_ttft_s"] is not None else "–"
    mturns = t["median_turns"] if t["median_turns"] is not None else "–"
    return f"""<h1>agent-serving-bench &mdash; run report</h1>
<div class="sub">Pi coding-agent trajectories on SWE-PolyBench &middot; harness held fixed</div>
<div class="kpis">
  <div class="kpi"><div class="n">{t["tasks"]}</div><div class="l">tasks</div></div>
  <div class="kpi"><div class="n">{t["submitted"]}</div><div class="l">submitted</div></div>
  <div class="kpi"><div class="n">{t["resolved"]}/{t["verified"]}</div><div class="l">resolved</div></div>
  <div class="kpi"><div class="n">{rr}</div><div class="l">resolve rate</div></div>
  <div class="kpi"><div class="n">{mttft}s</div><div class="l">median TTFT</div></div>
  <div class="kpi"><div class="n">{mturns}</div><div class="l">median turns</div></div>
</div>
{cards}"""


def build_html(rows, summary):
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>agent-serving-bench report</title>'
            f'<style>{CSS}</style></head><body>\n{render_body(rows, summary)}\n</body></html>')


def build_artifact(rows, summary):
    """Body-only fragment for the Artifact tool (it supplies <head>/<body>)."""
    return f'<style>{CSS}</style>\n{render_body(rows, summary)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="print file:// URL to open")
    ap.add_argument("--artifact", metavar="PATH",
                    help="also write a body-only HTML fragment for the Artifact tool")
    args = ap.parse_args()

    rows = collect()
    print_table(rows)

    summary = build_summary(rows)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    report = RESULTS / "report.html"
    report.write_text(build_html(rows, summary))
    print(f"\nwrote {RESULTS/'summary.json'}\nwrote {report}")
    if args.artifact:
        pathlib.Path(args.artifact).write_text(build_artifact(rows, summary))
        print(f"wrote {args.artifact}")
    if args.open:
        print(f"open: file://{report.resolve()}")


if __name__ == "__main__":
    main()
