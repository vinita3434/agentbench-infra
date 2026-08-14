#!/usr/bin/env python3
"""Build one detailed, shareable "proof of work" artifact PER TASK.

Unlike report.py (which rolls every task into one dashboard), this emits a
self-contained page for a single results/<model>/<task> run:

  - the verdict (resolved / not) as a bold banner
  - detailed metrics in clear labelled segments
  - the F2P / P2P tests, each named, pass/fail
  - a TTFT-per-turn chart
  - the trajectory as a compact list: one line per turn (what it did)

Output is a body-only HTML fragment (the Artifact tool supplies <head>/<body>).

Usage:
  task_artifact.py <task_dir> <out.html>
  task_artifact.py --all <out_dir>        # one file per results task
"""
import html
import json
import pathlib
import statistics
import sys

import extract_metrics  # sibling: metrics_for(), ttft data

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
TASKS = REPO_ROOT / "tasks" / "data"

HARNESS_BLURB = (
    "Pi coding agent (@earendil-works/pi-coding-agent), unmodified. An autonomous "
    "agentic loop: given only the issue text, it explores the repo, reasons, edits "
    "files, and runs shell/test commands via tool calls. The model is swapped behind "
    "an OpenAI-compatible endpoint — held fixed across this benchmark so results "
    "reflect the model, not the harness."
)


def load_task(task_id):
    f = TASKS / f"{task_id}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def difficulty(task):
    """Derived difficulty proxy from the gold patch's scope + hidden test count.

    Not an official SWE-PolyBench label. num_nodes = code nodes the fix touches;
    F2P = number of tests that must flip fail->pass.
    """
    import ast as _ast
    nodes = task.get("num_nodes") or 0
    try:
        f2p = len(_ast.literal_eval(task.get("F2P") or "[]"))
    except (ValueError, SyntaxError):
        f2p = 0
    single = task.get("is_single_func") or task.get("is_single_class")
    if (single and nodes <= 2 and f2p <= 2) or (nodes <= 1 and f2p <= 1):
        return "Easy"
    if nodes >= 6 or f2p >= 4:
        return "Hard"
    return "Medium"

CSS = """
  :root { color-scheme: light dark;
    --bg:#fbfbfd; --fg:#17181f; --mut:#6a6c7a; --line:#e6e6ef; --card:#ffffff;
    --ink:#0f1016; --ok:#15803d; --okbg:#dcfce7; --bad:#b91c1c; --badbg:#fee2e2;
    --warn:#b45309; --accent:#4f46e5; --accent2:#7c74f0; --code:#f4f4fb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0d0f14; --fg:#e7e8ee; --mut:#9a9db0; --line:#23262f; --card:#151822;
      --ink:#f3f4f8; --ok:#4ade80; --okbg:#0f2a1a; --bad:#f87171; --badbg:#2a1214;
      --warn:#fbbf24; --accent:#8b8cf6; --accent2:#a5a3f7; --code:#12141c; } }
  :root[data-theme="dark"] { --bg:#0d0f14; --fg:#e7e8ee; --mut:#9a9db0; --line:#23262f;
    --card:#151822; --ink:#f3f4f8; --ok:#4ade80; --okbg:#0f2a1a; --bad:#f87171;
    --badbg:#2a1214; --warn:#fbbf24; --accent:#8b8cf6; --accent2:#a5a3f7; --code:#12141c; }
  :root[data-theme="light"] { --bg:#fbfbfd; --fg:#17181f; --mut:#6a6c7a; --line:#e6e6ef;
    --card:#ffffff; --ink:#0f1016; --ok:#15803d; --okbg:#dcfce7; --bad:#b91c1c;
    --badbg:#fee2e2; --warn:#b45309; --accent:#4f46e5; --accent2:#7c74f0; --code:#f4f4fb; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.6 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .doc { max-width:860px; margin:0 auto; padding:40px 24px 72px; }
  .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:700;
    color:var(--accent); margin-bottom:10px; }
  h1 { font-size:30px; line-height:1.15; margin:0 0 6px; color:var(--ink);
    font-family:ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:-.01em; text-wrap:balance; }
  .sub { color:var(--mut); font-size:13.5px; margin-bottom:22px; }
  .sub b { color:var(--fg); }
  .verdict { display:inline-flex; align-items:center; gap:10px; font-weight:800;
    font-size:15px; padding:10px 18px; border-radius:12px; margin-bottom:34px; letter-spacing:.02em; }
  .verdict.ok { background:var(--okbg); color:var(--ok); }
  .verdict.bad { background:var(--badbg); color:var(--bad); }
  .verdict.warn { background:color-mix(in srgb, var(--warn) 16%, transparent); color:var(--warn); }
  .verdict .dot { width:9px; height:9px; border-radius:50%; background:currentColor; }
  .errnote { background:color-mix(in srgb, var(--warn) 9%, transparent);
    border:1px solid color-mix(in srgb, var(--warn) 35%, transparent); border-radius:12px;
    padding:12px 16px; margin:-20px 0 32px; font-size:13px; color:var(--fg); }
  .errnote b { color:var(--warn); }
  .about { display:grid; grid-template-columns:120px 1fr; gap:2px 16px;
    background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 20px; }
  .about .ak { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--mut);
    font-weight:700; padding-top:3px; }
  .about .av { font-size:14px; color:var(--fg); }
  .about .av.mono { font-family:ui-monospace,Menlo,monospace; font-size:13px; }
  .about .blurb { color:var(--mut); font-size:13px; line-height:1.55; }
  .pill { display:inline-block; padding:2px 10px; border-radius:20px; font-size:12px;
    font-weight:700; background:var(--code); }
  .pill.easy { color:var(--ok); background:var(--okbg); }
  .pill.hard { color:var(--bad); background:var(--badbg); }
  .pill.medium { color:var(--warn); background:color-mix(in srgb, var(--warn) 15%, transparent); }
  .pill.type { color:var(--accent); background:color-mix(in srgb, var(--accent) 14%, transparent); }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.1em; color:var(--mut);
    font-weight:700; margin:36px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line); }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--line);
    border:1px solid var(--line); border-radius:14px; overflow:hidden; }
  @media (max-width:640px){ .stats { grid-template-columns:repeat(2,1fr); } }
  .stat { background:var(--card); padding:16px 18px; }
  .stat .l { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); }
  .stat .n { font-size:24px; font-weight:800; color:var(--ink); margin-top:4px;
    font-variant-numeric:tabular-nums; }
  .stat .n.good { color:var(--ok); } .stat .n.bad { color:var(--bad); }
  .stat .u { font-size:13px; font-weight:600; color:var(--mut); margin-left:3px; }
  .tests { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:640px){ .tests { grid-template-columns:1fr; } }
  .tests h3 { font-size:13px; margin:0 0 8px; color:var(--ink); }
  .tests ul { list-style:none; margin:0; padding:0; font-size:13px; }
  .tests li { padding:5px 0; border-bottom:1px dashed var(--line); display:flex; gap:8px;
    align-items:flex-start; overflow-wrap:anywhere; }
  .tpass .mk { color:var(--ok); } .tfail .mk { color:var(--bad); } .mk { font-weight:800; }
  .spark { width:100%; border:1px solid var(--line); border-radius:12px; background:var(--card);
    display:block; }
  .spark polyline { fill:none; stroke:var(--accent); stroke-width:2; }
  .spark circle { fill:var(--accent); }
  .spark .end { fill:var(--accent2); }
  .axis { fill:var(--mut); font-size:10px; font-variant-numeric:tabular-nums; }
  ol.turns { list-style:none; margin:0; padding:0; counter-reset:t; }
  ol.turns li { display:grid; grid-template-columns:52px 62px 1fr; gap:12px; align-items:baseline;
    padding:7px 0; border-bottom:1px solid var(--line); font-size:13.5px; }
  ol.turns .tn { color:var(--mut); font-variant-numeric:tabular-nums; font-weight:600; }
  ol.turns .tt { color:var(--accent); font-variant-numeric:tabular-nums; font-size:12px; }
  ol.turns .what { color:var(--fg); overflow-wrap:anywhere; }
  ol.turns .what code { background:var(--code); padding:1px 6px; border-radius:5px;
    font:12.5px ui-monospace,Menlo,monospace; }
  ol.turns li.slow .tt { color:var(--warn); font-weight:700; }
  .brief { background:var(--card); border:1px solid var(--line); border-left:3px solid var(--accent);
    border-radius:12px; padding:16px 20px; }
  .brief .brieft { font-size:16px; font-weight:700; color:var(--ink); line-height:1.35;
    text-wrap:balance; }
  .brief .briefb { margin:8px 0 0; font-size:14px; line-height:1.6; color:var(--fg);
    max-width:62ch; }
  .brief .briefnote { margin-top:12px; font-size:11.5px; color:var(--mut);
    text-transform:uppercase; letter-spacing:.05em; }
  .diffgrid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:760px){ .diffgrid { grid-template-columns:1fr; } }
  .diffwrap { border:1px solid var(--line); border-radius:12px; overflow:hidden;
    background:var(--card); min-width:0; }
  .difflabel { font-size:11px; text-transform:uppercase; letter-spacing:.07em; font-weight:700;
    color:var(--mut); padding:10px 14px; border-bottom:1px solid var(--line); background:var(--code); }
  .diff { overflow-x:auto; padding:10px 0; font:12.5px/1.65 ui-monospace,Menlo,monospace; }
  .dl { padding:0 14px; white-space:pre; }
  .dl.dh { color:var(--mut); }
  .dl.dhunk { color:var(--accent); background:color-mix(in srgb, var(--accent) 8%, transparent); }
  .dl.dadd { color:var(--ok); background:color-mix(in srgb, var(--ok) 12%, transparent); }
  .dl.ddel { color:var(--bad); background:color-mix(in srgb, var(--bad) 12%, transparent); }
  .foot { margin-top:34px; color:var(--mut); font-size:12px; }
"""


def esc(s):
    return html.escape(str(s if s is not None else ""))


def median(xs):
    return round(statistics.median(xs), 2) if xs else None


def ttft_svg(ttft, w=812, h=140, pad=26):
    if not ttft:
        return '<div class="sub">No TTFT captured (native run without stream timing).</div>'
    if len(ttft) == 1:
        return f'<div class="sub">TTFT {ttft[0]}s (single turn).</div>'
    lo, hi = min(ttft), max(ttft)
    span = (hi - lo) or 1.0
    n = len(ttft)
    pts = []
    for i, v in enumerate(ttft):
        x = pad + (w - 2 * pad) * i / (n - 1)
        y = h - pad - (h - 2 * pad) * (v - lo) / span
        pts.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(
        f'<circle class="{"end" if i in (0, n-1) else ""}" cx="{x:.1f}" cy="{y:.1f}" r="{3 if i in (0,n-1) else 2}"/>'
        for i, (x, y) in enumerate(pts))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" role="img" aria-label="TTFT per turn">'
            f'<polyline points="{poly}"/>{dots}'
            f'<text class="axis" x="{pad}" y="16">{hi}s peak</text>'
            f'<text class="axis" x="{pad}" y="{h-6}">{lo}s low</text>'
            f'<text class="axis" x="{w-pad-70}" y="{h-6}">turn 1 &rarr; {n}</text></svg>')


def turn_oneline(t):
    """One human line describing what the turn did."""
    acts = t["actions"]
    if acts:
        a = acts[0]
        arg = f' <code>{esc(a["args"])}</code>' if a["args"] else ""
        extra = f' <span style="color:var(--mut)">(+{len(acts)-1} more)</span>' if len(acts) > 1 else ""
        verb = {"read": "read", "edit": "edited", "write": "wrote",
                "bash": "ran", "grep": "searched"}.get(a["tool"], a["tool"])
        return f'{verb}{arg}{extra}'
    r = (t.get("reasoning") or "").strip()
    if r:
        return f'<span style="color:var(--mut)">thinking:</span> {esc(r[:110])}'
    return '<span style="color:var(--mut)">final response</span>'


ISSUE_NOISE = (
    "### system info", "### who can help", "### information", "### related",
    "### expected behavior", "### reproduction", "### environment",
    "who can help", "system info",
    # Issue-template instructions to the *reporter*, not description of the bug.
    "tip!", "don't write this stuff", "do not write this stuff",
    "go to https://", "paste your code", "press the", "playground link",
)


def _is_noise_line(line):
    """Filter a line of issue prose down to something worth showing.

    Real GitHub issues carry three kinds of padding that read badly in a
    summary: environment bullets ("- Python: 3.10"), template instructions to
    the reporter, and enormous shareable URLs (prettier's playground encodes
    the whole snippet into the link).
    """
    low = line.lower()
    if any(low.startswith(n) or n in low[:40] for n in ISSUE_NOISE):
        return True
    # Tail of an HTML comment whose opener was already skipped.
    if line.startswith("-->"):
        return True
    # A line that is only a GitHub @mention ("@vowelparrot").
    if line.startswith("@") and len(line) < 40 and " " not in line:
        return True
    # Long unbroken tokens are URLs / encoded payloads, never prose.
    if any(len(tok) > 60 for tok in line.split()):
        return True
    # Environment bullets: short list items of the form "- Key: value".
    if line[:2] in ("- ", "* ") and ":" in line and len(line) < 60:
        return True
    # Bold-only scaffolding labels: "**Input:**", "**Prettier 1.8.2**".
    if line.startswith("**") and line.endswith("**") and len(line) < 40:
        return True
    return False


def task_summary(task):
    """(headline, blurb) describing the bug, taken from the issue text.

    SWE-PolyBench problem statements are raw GitHub issues: a title line, then
    a body padded with template scaffolding (checkbox lists, "Who can help?",
    version tables). Pull the title and the first substantive prose so the
    artifact says what was actually wrong, without pasting 20KB of template.
    """
    text = (task.get("problem_statement") or "").strip()
    if not text:
        return "", ""

    lines = text.splitlines()
    headline = ""
    body_start = 0
    for i, raw in enumerate(lines):
        line = raw.strip().lstrip("#").strip()
        if line:
            headline = line
            body_start = i + 1
            break

    keep = []
    for raw in lines[body_start:]:
        line = raw.strip()
        low = line.lower()
        if not line:
            continue
        if line.startswith(("- [", "* [", "```", "|", ">", "<!--", "#")):
            continue          # checkboxes, code fences, tables, quotes, headers
        if _is_noise_line(line):
            continue
        keep.append(line)
        if sum(len(k) for k in keep) > 420:
            break

    blurb = " ".join(keep)
    if len(blurb) > 400:
        cut = blurb[:400]
        # Prefer a sentence boundary so the summary doesn't end mid-word.
        dot = cut.rfind(". ")
        blurb = (cut[:dot + 1] if dot > 220 else cut.rstrip() + "…")
    return headline[:160], blurb


def diff_block(text, label, cls=""):
    """One unified diff, line-coloured. Hunk headers dim, +/- carry the colour."""
    if not (text or "").strip():
        return ""
    lines = []
    for ln in text.splitlines():
        kind = ""
        if ln.startswith("+++") or ln.startswith("---") or ln.startswith("diff ") \
                or ln.startswith("index "):
            kind = "dh"          # file header: present but recessive
        elif ln.startswith("@@"):
            kind = "dhunk"
        elif ln.startswith("+"):
            kind = "dadd"
        elif ln.startswith("-"):
            kind = "ddel"
        lines.append(f'<div class="dl {kind}">{esc(ln) or "&nbsp;"}</div>')
    return (f'<div class="diffwrap {cls}"><div class="difflabel">{esc(label)}</div>'
            f'<div class="diff">{"".join(lines)}</div></div>')


def patch_html(task_dir, task):
    """The change the agent actually made, beside the reference fix.

    The verdict says whether the tests passed; this says what was written. Both
    are shown because they answer different questions -- a task can pass with a
    patch that looks nothing like gold, and that difference is the interesting
    part of a trajectory, not a defect.
    """
    cand = (task_dir / "candidate.patch")
    cand_text = cand.read_text() if cand.exists() else ""
    if not cand_text.strip():
        return ('\n  <h2>Patch produced</h2>\n  <div class="sub">'
                'No patch was produced &mdash; the agent finished without editing '
                'any file.</div>')
    gold = task.get("patch") or ""
    return (f'\n  <h2>Patch produced</h2>\n  <div class="diffgrid">'
            f'{diff_block(cand_text, "candidate · written by the agent")}'
            f'{diff_block(gold, "reference · the dataset’s own fix")}'
            f'</div>')


def build_fragment(task_dir):
    task_dir = pathlib.Path(task_dir)
    m = extract_metrics.metrics_for(task_dir)
    (task_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    rr = {}
    if (task_dir / "run_record.json").exists():
        rr = json.loads((task_dir / "run_record.json").read_text())
    v = None
    if (task_dir / "verify.json").exists():
        v = json.loads((task_dir / "verify.json").read_text())

    task = load_task(m["task_id"])
    diff = difficulty(task)
    category = task.get("task_category") or "—"
    language = task.get("language") or "—"
    deployment = rr.get("deployment") or "docker"

    vstatus = v.get("status") if v else None
    verr = v.get("error") if v else None
    resolved = v["resolved"] if v else None
    errnote = ""
    if vstatus == "resolved":
        verdict = '<div class="verdict ok"><span class="dot"></span>RESOLVED</div>'
    elif vstatus == "not_resolved":
        verdict = '<div class="verdict bad"><span class="dot"></span>NOT RESOLVED</div>'
    elif vstatus == "error":
        verdict = '<div class="verdict warn"><span class="dot"></span>COULD NOT VERIFY</div>'
        errnote = f'<div class="errnote"><b>Verification did not run.</b> {esc(verr)}</div>'
    else:
        verdict = ('<div class="verdict warn"><span class="dot"></span>VERIFICATION PENDING</div>')

    ttft = m["ttft_trajectory"]
    tok = m["tokens"]
    tokstr = f'{tok["input"]}/{tok["output"]}' if tok["available"] else "n/a"
    wall = m["wall_seconds"]
    wall_str = f"{wall//60}m {wall%60}s" if isinstance(wall, int) else esc(wall)

    def stat(label, n, unit="", cls=""):
        u = f'<span class="u">{unit}</span>' if unit else ""
        return f'<div class="stat"><div class="l">{label}</div><div class="n {cls}">{n}{u}</div></div>'

    f2p = v["f2p"] if v else None
    p2p = v["p2p"] if v else None
    err = vstatus == "error"
    if err:
        resolved_cell, resolved_cls = "n/a", ""
        f2p_cell, f2p_cls = "not run", ""
        p2p_cell, p2p_cls = "not run", ""
    else:
        resolved_cell = ("yes" if resolved else "no") if v else "–"
        resolved_cls = "good" if resolved else ("bad" if v else "")
        f2p_cell = f'{f2p["passed"]}/{f2p["total"]}' if f2p else "–"
        f2p_cls = "good" if (f2p and f2p["passed"] == f2p["total"]) else ("bad" if f2p else "")
        p2p_cell = f'{p2p["passed"]}/{p2p["total"]}' if p2p else "–"
        p2p_cls = "good" if (p2p and p2p["passed"] == p2p["total"]) else ""
    stats = "".join([
        stat("Submitted", "yes" if m["submitted"] else "no", cls="good" if m["submitted"] else "bad"),
        stat("Resolved", resolved_cell, cls=resolved_cls),
        stat("F2P passed", f2p_cell, cls=f2p_cls),
        stat("P2P passed", p2p_cell, cls=p2p_cls),
        stat("Turns", m["turns"]),
        stat("Turns to submit", esc(m["turns_to_submit"])),
        stat("Median TTFT", median(ttft) if ttft else "–", "s"),
        stat("Wall time", wall_str),
    ])

    def testlist(grp):
        if not grp:
            return '<div class="sub">no data</div>'
        items = "".join(
            f'<li class="{"tpass" if t["passed"] else "tfail"}">'
            f'<span class="mk">{"✓" if t["passed"] else "✗"}</span>'
            f'<span>{esc(t["test"].split("->")[-1])}</span></li>'
            for t in grp["tests"])
        return f'<ul>{items}</ul>'

    tests_html = ""
    if err:
        tests_html = ('\n    <h2>Hidden tests</h2>\n    <div class="sub">'
                      f'Not run &mdash; {esc(f2p["total"] if f2p else "?")} F2P + '
                      f'{esc(p2p["total"] if p2p else "?")} P2P tests could not execute in this environment '
                      '(see note above). Verify under Mode B.</div>')
    elif v:
        tests_html = f"""
    <h2>Hidden tests</h2>
    <div class="tests">
      <div><h3>Fail&#8209;to&#8209;Pass &mdash; {f2p["passed"]}/{f2p["total"]} ({f2p["pct"]}%)</h3>{testlist(f2p)}</div>
      <div><h3>Pass&#8209;to&#8209;Pass &mdash; {p2p["passed"]}/{p2p["total"]} ({p2p["pct"]}%)</h3>{testlist(p2p)}</div>
    </div>"""

    # one line per turn
    slow_cut = max(ttft) if ttft else None
    turn_lis = ""
    for pt in m["per_turn"]:
        tt = pt["ttft_s"]
        slow = " slow" if (pt["duration_s"] is not None and pt["duration_s"] > 120) else ""
        tt_str = f'{tt}s' if tt is not None else '–'
        turn_lis += (f'<li class="{slow.strip()}"><span class="tn">turn {pt["turn"]}</span>'
                     f'<span class="tt">{tt_str}</span>'
                     f'<span class="what">{turn_oneline(pt)}</span></li>')

    headline, blurb = task_summary(task)
    summary_html = ""
    if headline:
        summary_html = (
            '\n  <h2>The task</h2>\n  <div class="brief">'
            f'<div class="brieft">{esc(headline)}</div>'
            + (f'<p class="briefb">{esc(blurb)}</p>' if blurb else "")
            + '<div class="briefnote">Reported issue text, as given to the agent — '
              'no repository context, no test names, no hints.</div></div>')

    base = (rr.get("base_commit") or "")[:10]
    return f"""<style>{CSS}</style>
<div class="doc">
  <div class="eyebrow">agent-serving-bench &middot; proof of work</div>
  <h1>{esc(m["task_id"])}</h1>
  <div class="sub">{esc(rr.get("timestamp_utc"))} &middot; run on <b>{esc(deployment)}</b></div>
  {verdict}
  {errnote}
  {summary_html}

  <h2>About this run</h2>
  <div class="about">
    <div class="ak">Model</div><div class="av mono">{esc(m["model"])} <span style="color:var(--mut)">via {esc(rr.get("provider"))}</span></div>
    <div class="ak">Harness</div><div class="av blurb">{HARNESS_BLURB}</div>
    <div class="ak">Task</div><div class="av mono">{esc(m["task_id"])} <span style="color:var(--mut)">&middot; {esc(rr.get("repo"))} @ {esc(base)}</span></div>
    <div class="ak">Type</div><div class="av"><span class="pill type">{esc(category)}</span> <span style="color:var(--mut)">&middot; {esc(language)}</span></div>
    <div class="ak">Difficulty</div><div class="av"><span class="pill {diff.lower()}">{esc(diff)}</span> <span style="color:var(--mut)">&middot; derived from {esc(task.get("num_nodes"))} changed node(s), {esc(f2p["total"] if f2p else "?")} F2P test(s)</span></div>
    <div class="ak">Deployment</div><div class="av">{esc(deployment)} <span style="color:var(--mut)">&middot; Mode {"A (Docker / OpenRouter)" if deployment=="docker" else "B (RunPod / SGLang)"}</span></div>
  </div>

  <h2>Metrics</h2>
  <div class="stats">{stats}</div>
  {tests_html}

  <h2>Time to first token &middot; per turn</h2>
  {ttft_svg(ttft)}

  <h2>Trajectory &middot; {m["turns"]} turns</h2>
  <ol class="turns">{turn_lis}</ol>
  {patch_html(task_dir, task)}

  <div class="foot">Pi coding agent (unmodified) on SWE-PolyBench &middot; harness held fixed &middot;
    tokens in/out: {tokstr} &middot; pi exit {esc(rr.get("pi_exit_code"))}</div>
</div>"""


def out_path_for(task_dir):
    """results/artifacts/<deployment>/<model>/<task>.html — deployment from run_record.

    The model segment is required: two models on the same task would otherwise
    write to the same file and the second would silently overwrite the first.
    """
    task_dir = pathlib.Path(task_dir)
    rr = {}
    if (task_dir / "run_record.json").exists():
        rr = json.loads((task_dir / "run_record.json").read_text())
    dep = rr.get("deployment") or "docker"
    d = RESULTS / "artifacts" / dep / task_dir.parent.name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{task_dir.name}.html"


def generate(task_dir):
    out = out_path_for(task_dir)
    out.write_text(build_fragment(task_dir))
    print(f"wrote {out}")
    return out


def main():
    args = sys.argv[1:]
    if args and args[0] == "--all":
        for model_dir in sorted(RESULTS.glob("*")):
            if not model_dir.is_dir() or model_dir.name in ("serve", "_gold", "artifacts"):
                continue
            for td in sorted(model_dir.iterdir()):
                if td.is_dir() and (td / "run_record.json").exists():
                    generate(td)
        return
    if len(args) == 1:                       # auto-route into artifacts/<deployment>/
        generate(args[0])
    elif len(args) == 2:                     # explicit output path (back-compat)
        pathlib.Path(args[1]).write_text(build_fragment(args[0]))
        print(f"wrote {args[1]}")
    else:
        sys.exit("usage: task_artifact.py <task_dir> [out.html]  |  --all")


if __name__ == "__main__":
    main()
