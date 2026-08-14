#!/usr/bin/env python3
"""Parse a task's test run into per-test F2P/P2P results.

Multi-language: the task's test_command emits results either as jest --json (JS/TS)
on stdout, or a JUnit XML file (pytest / Maven-surefire for Python/Java). This
picks whichever is available.

Crucially it distinguishes three outcomes:
  - resolved / not-resolved  (tests actually ran and were parsed)
  - ERROR / could-not-verify  (the test process never produced results, e.g. it
    crashed — TensorFlow under x86 emulation raises SIGILL "Illegal instruction";
    this is NOT a model failure and must not be scored 0/N)

Resolved (strict) = ALL F2P pass AND ALL P2P pass.

Usage:
  _verify_parse.py <task_json> <stdout> <out_json> <patch_source> \
                   <model_apply> <test_apply> [<stderr>] [<junit_xml>]
"""
import ast
import json
import sys
import xml.etree.ElementTree as ET

CRASH_SIGNS = ("Illegal instruction", "Segmentation fault", "core dumped",
               "qemu: uncaught", "cannot execute binary", "Bus error",
               "Fatal Python error", "std::bad_alloc")


def load_list(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []


# --- jest (JS/TS) ---------------------------------------------------------
def extract_jest_json(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"testResults"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    dec = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            return None
        try:
            obj, _ = dec.raw_decode(text, start)
            if isinstance(obj, dict) and "testResults" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        idx = start + 1


def jest_passed_map(jest):
    passed = {}
    for fileres in jest.get("testResults", []):
        fname = fileres.get("name", "")
        for a in fileres.get("assertionResults", []):
            ok = a.get("status") == "passed"
            names = {a.get("fullName"), a.get("title")}
            anc = a.get("ancestorTitles") or []
            if anc and a.get("title"):
                names.add(" ".join(anc + [a["title"]]))
            for n in names:
                if n:
                    passed[(fname, n)] = ok
    return passed


def check_jest(entries, passed):
    out = []
    for e in entries:
        fpath, tname = (e.split("->", 1) if "->" in e else ("", e))
        key = (fpath, tname)
        if key in passed:
            ok = passed[key]
        else:
            hits = [v for (f, n), v in passed.items() if n == tname]
            ok = all(hits) if hits else None
        out.append({"test": e, "status": ("passed" if ok else "failed") if ok is not None else "not_found",
                    "passed": bool(ok)})
    return out


# --- mocha --reporter json (JS, e.g. serverless) --------------------------
def extract_mocha_json(text):
    dec = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            return None
        try:
            obj, _ = dec.raw_decode(text, start)
            if isinstance(obj, dict) and "stats" in obj and \
               any(k in obj for k in ("passes", "failures", "tests")):
                return obj
        except json.JSONDecodeError:
            pass
        idx = start + 1


def mocha_passed_map(mocha):
    """fullTitle -> passed. passes=True, failures/pending=False."""
    passed = {}
    for t in mocha.get("failures", []):
        if t.get("fullTitle"):
            passed[t["fullTitle"]] = False
    for t in mocha.get("pending", []):
        if t.get("fullTitle"):
            passed.setdefault(t["fullTitle"], False)
    for t in mocha.get("passes", []):
        if t.get("fullTitle"):
            passed[t["fullTitle"]] = True
    # fallback: derive from `tests` (err present == failed)
    if not passed:
        for t in mocha.get("tests", []):
            if t.get("fullTitle"):
                passed[t["fullTitle"]] = not (t.get("err") or {})
    return passed


def check_by_name(entries, passed):
    out = []
    for e in entries:
        if e in passed:
            ok = passed[e]
        else:
            hits = [v for k, v in passed.items() if k == e or k.endswith(e) or e.endswith(k)]
            ok = all(hits) if hits else None
        out.append({"test": e, "status": ("passed" if ok else "failed") if ok is not None else "not_found",
                    "passed": bool(ok)})
    return out


# --- JUnit XML (pytest / surefire) ---------------------------------------
def parse_junit(path):
    """Return list of (classname, name, passed) from a JUnit XML file."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    cases = []
    for tc in root.iter("testcase"):
        cls = tc.get("classname", "") or ""
        name = tc.get("name", "") or ""
        bad = any(tc.iter(tag) and next(tc.iter(tag), None) is not None
                  for tag in ("failure", "error"))
        # skipped counts as not-passed
        skipped = next(tc.iter("skipped"), None) is not None
        cases.append((cls, name, not bad and not skipped))
    return cases


# --- pytest --json-report (.report.json) ---------------------------------
def parse_json_report(path):
    """{nodeid: passed} from pytest-json-report's .report.json.

    A test counts as passed only if every phase that ran (setup/call/teardown)
    passed -- an erroring teardown still means the test did not cleanly pass.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    tests = data.get("tests")
    if not isinstance(tests, list):
        return None
    out = {}
    for t in tests:
        nodeid = t.get("nodeid")
        if not nodeid:
            continue
        phases = [t.get(p, {}).get("outcome") for p in ("setup", "call", "teardown")]
        phases = [p for p in phases if p]
        out[nodeid] = (t.get("outcome") == "passed"
                       and all(p == "passed" for p in phases))
    return out or None


def check_json_report(entries, passed):
    """Match SWE-PolyBench's 'file:Class:test' entries against pytest nodeids.

    The dataset writes 'path.py:None:test_name[param]' where pytest writes
    'path.py::test_name[param]', so compare on (file, test) after normalizing
    both sides rather than string-matching the whole id.
    """
    def split_nodeid(nid):
        parts = nid.split("::")
        return parts[0], parts[-1]

    norm = {}
    for nid, ok in passed.items():
        f, t = split_nodeid(nid)
        norm[(f, t)] = ok

    out = []
    for e in entries:
        bits = e.split(":")
        # 'file:Class:test' -- Class is literally 'None' for module-level tests.
        fname, test = (bits[0], bits[-1]) if len(bits) >= 2 else ("", e)
        hit = norm.get((fname, test))
        if hit is None:
            cands = [ok for (f, t), ok in norm.items()
                     if t == test and (not fname or f.endswith(fname) or fname.endswith(f))]
            hit = all(cands) if cands else None
        out.append({"test": e,
                    "status": ("passed" if hit else "failed") if hit is not None else "not_found",
                    "passed": bool(hit)})
    return out


def parse_junit_all(primary):
    """Cases from `primary`, plus any sibling surefire TEST-*.xml files.

    pytest writes one XML; Maven surefire writes one per test class, which
    verify_task.sh copies next to it. Merging them means a Java task is scored
    from its whole run rather than whichever file happened to be first.
    """
    import glob
    import os

    cases = []
    if primary and os.path.exists(primary):
        cases.extend(parse_junit(primary) or [])
    for extra in sorted(glob.glob(os.path.join(os.path.dirname(primary or "."),
                                               "TEST-*.xml"))):
        cases.extend(parse_junit(extra) or [])
    return cases or None


def check_junit(entries, cases):
    """Match 'file:Class:test' (pytest nodeid) against JUnit testcases."""
    out = []
    for e in entries:
        parts = e.split(":")
        test = parts[-1]
        klass = parts[-2] if len(parts) >= 2 else ""
        hit = None
        # 1) name + class substring match
        for cls, name, ok in cases:
            if name == test and (not klass or klass in cls):
                hit = ok
                break
        # 2) name-only fallback
        if hit is None:
            names = [ok for cls, name, ok in cases if name == test]
            if names:
                hit = all(names)
        out.append({"test": e, "status": ("passed" if hit else "failed") if hit is not None else "not_found",
                    "passed": bool(hit)})
    return out


def _grp(entries, results):
    ok = sum(1 for r in results if r["passed"])
    return {"passed": ok, "total": len(entries),
            "pct": round(100 * ok / len(entries), 1) if entries else 0.0,
            "tests": results}


def _notrun(entries):
    return {"passed": 0, "total": len(entries), "pct": 0.0,
            "tests": [{"test": t, "status": "not_run", "passed": False} for t in entries]}


def main():
    a = sys.argv[1:]
    task_json, stdout_f, out_json, patch_source, model_apply, test_apply = a[:6]
    stderr_f = a[6] if len(a) > 6 else None
    junit_f = a[7] if len(a) > 7 else None
    jsonrep_f = a[8] if len(a) > 8 else None

    task = json.load(open(task_json))
    f2p = load_list(task.get("F2P"))
    p2p = load_list(task.get("P2P"))
    stdout = open(stdout_f, errors="replace").read() if stdout_f else ""
    stderr = open(stderr_f, errors="replace").read() if stderr_f and __import__("os").path.exists(stderr_f) else ""

    verify = {
        "instance_id": task.get("instance_id"),
        "patch_source": patch_source,
        "model_patch_apply": model_apply,
        "test_patch_apply": test_apply,
    }

    # Pick a results source.
    jest = extract_jest_json(stdout) if stdout else None
    mocha = extract_mocha_json(stdout) if (stdout and jest is None) else None
    junit = parse_junit_all(junit_f) if junit_f else None
    jsonrep = (parse_json_report(jsonrep_f)
               if jsonrep_f and __import__("os").path.exists(jsonrep_f) else None)

    if jest is not None:
        pm = jest_passed_map(jest)
        f2p_res, p2p_res = check_jest(f2p, pm), check_jest(p2p, pm)
        verify["result_source"] = "jest"
    elif mocha is not None:
        pm = mocha_passed_map(mocha)
        f2p_res, p2p_res = check_by_name(f2p, pm), check_by_name(p2p, pm)
        verify["result_source"] = "mocha"
    elif junit:
        f2p_res, p2p_res = check_junit(f2p, junit), check_junit(p2p, junit)
        verify["result_source"] = "junit"
    elif jsonrep:
        f2p_res = check_json_report(f2p, jsonrep)
        p2p_res = check_json_report(p2p, jsonrep)
        verify["result_source"] = "pytest-json-report"
    else:
        # No parseable results. Was it a crash / env error, or a clean no-run?
        crash = next((s for s in CRASH_SIGNS if s in stderr), None)
        verify.update({
            "resolved": None,
            "status": "error",
            "error": (f"test process did not produce results: '{crash}'. "
                      "Likely x86-emulation incompatibility (e.g. TensorFlow SIGILL) "
                      "— run this task under Mode B (native x86 / RunPod)."
                      if crash else
                      "test_command produced no parseable results "
                      "(no jest/mocha json, JUnit xml, or pytest json-report)."),
            "f2p": _notrun(f2p),
            "p2p": _notrun(p2p),
        })
        json.dump(verify, open(out_json, "w"), indent=2)
        _print(verify)
        return

    f2p_g, p2p_g = _grp(f2p, f2p_res), _grp(p2p, p2p_res)
    resolved = (len(f2p) > 0 and f2p_g["passed"] == len(f2p)
                and p2p_g["passed"] == len(p2p)
                and model_apply == "OK" and test_apply == "OK")
    verify.update({"resolved": resolved,
                   "status": "resolved" if resolved else "not_resolved",
                   "f2p": f2p_g, "p2p": p2p_g})
    json.dump(verify, open(out_json, "w"), indent=2)
    _print(verify)


def _print(v):
    st = v.get("status")
    if st == "error":
        print(f"[verify] {v['instance_id']} [{v['patch_source']}]: COULD NOT VERIFY")
        print(f"[verify]   {v['error']}")
        return
    r = "RESOLVED" if v.get("resolved") else "NOT resolved"
    f, p = v["f2p"], v["p2p"]
    print(f"[verify] {v['instance_id']} [{v['patch_source']}]: {r}  (src={v.get('result_source')})")
    print(f"[verify]   apply: model={v['model_patch_apply']} test={v['test_patch_apply']}")
    print(f"[verify]   F2P: {f['passed']}/{f['total']} ({f['pct']}%)   "
          f"P2P: {p['passed']}/{p['total']} ({p['pct']}%)")


if __name__ == "__main__":
    main()
