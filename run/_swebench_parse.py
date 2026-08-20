#!/usr/bin/env python3
"""Parse django's unittest output into per-test F2P/P2P results.

Django's suite runs under tests/runtests.py, which emits no JUnit xml -- so the
shared JUnit path in run/_verify_parse.py has nothing to read. It does print
unittest's verbose format, and SWE-bench's test ids are written in exactly that
format:

    test_available_apps (admin_views.test_adminsite.SiteEachContextTest) ... ok

so the id IS the line prefix. Newer django repeats the method name inside the
parens and may append a docstring; both are normalised away.

Kept separate from _verify_parse.py so the SWE-PolyBench path is untouched.

Usage:
  _swebench_parse.py <task_json> <stdout> <out_json> <patch_source>
                     <model_apply> <test_apply> [<stderr>]
"""
import json
import re
import sys

# "test_x (a.b.C)" or "test_x (a.b.C.test_x)", optional docstring, then the
# verdict -- which unittest may put on a later line when output interleaves.
LINE = re.compile(
    r"^(?P<name>\w+)\s+\((?P<ctx>[\w.]+)\)(?:\s+\S.*?)?\s*\.\.\.\s*"
    r"(?P<verdict>ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)?\s*$"
)
CRASH_SIGNS = ("Illegal instruction", "Segmentation fault", "core dumped",
               "Fatal Python error", "cannot execute binary")


def normalise(name, ctx):
    """(test_name, class_path) with a duplicated method name stripped."""
    parts = ctx.split(".")
    if parts and parts[-1] == name:
        parts = parts[:-1]
    return name, ".".join(parts)


ID_ONLY = re.compile(r"^(?P<name>\w+)\s+\((?P<ctx>[\w.]+)\)$")
DOC_VERDICT = re.compile(
    r"^(?P<doc>.*?)\s*\.\.\.\s*"
    r"(?P<verdict>ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)\s*$")


def parse_django(text):
    """Verdicts keyed BOTH by (test, class) and by docstring.

    unittest with descriptions on prints a test with a docstring over two lines:

        test_get_action (admin_views.test_adminsite.SiteActionsTests)
        AdminSite.get_action() returns an action even if disabled. ... ok

    SWE-bench's own scraper captured whichever line carried the verdict, so
    PASS_TO_PASS contains a mix of ids and bare docstrings. Indexing both is
    the only way to match every entry; keying on ids alone leaves docstring
    entries "not_found", which reads as a failure and sinks a correct patch.
    """
    by_id, by_doc = {}, {}
    pending = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        m = LINE.match(line)
        if m and m.group("verdict"):
            key = normalise(m.group("name"), m.group("ctx"))
            # skipped counts as not-passed: the graded behaviour was never
            # demonstrated, and calling that a pass inflates the resolve rate.
            by_id[key] = m.group("verdict").strip() == "ok"
            pending = None
            continue
        m = ID_ONLY.match(line)
        if m:
            pending = normalise(m.group("name"), m.group("ctx"))
            continue
        m = DOC_VERDICT.match(line)
        if m and pending:
            passed = m.group("verdict").strip() == "ok"
            by_id[pending] = passed
            doc = m.group("doc").strip()
            if doc:
                by_doc[doc] = passed
            pending = None
    return by_id, by_doc


def check(entries, passed, by_doc=None):
    by_doc = by_doc or {}
    results = []
    for entry in entries:
        entry = entry.strip()
        m = re.match(r"^(?P<name>\w+)\s+\((?P<ctx>[\w.]+)\)$", entry)
        hit = None
        if m:
            key = normalise(m.group("name"), m.group("ctx"))
            hit = passed.get(key)
            if hit is None:                       # fall back to name-only
                same = [v for (n, _), v in passed.items() if n == key[0]]
                hit = all(same) if same else None
        else:
            # Not an id -- treat it as a docstring, which is how SWE-bench
            # recorded tests that have one.
            hit = by_doc.get(entry)
            if hit is None:
                hit = by_doc.get(entry.rstrip("."))
        results.append({
            "test": entry,
            "status": ("passed" if hit else "failed") if hit is not None else "not_found",
            "passed": bool(hit),
        })
    return results


def group(entries, results):
    ok = sum(1 for r in results if r["passed"])
    return {"passed": ok, "total": len(entries),
            "pct": round(100 * ok / len(entries), 1) if entries else 0.0,
            "tests": results}


def main():
    a = sys.argv[1:]
    task_json, stdout_f, out_json, patch_source, model_apply, test_apply = a[:6]
    stderr_f = a[6] if len(a) > 6 else None

    task = json.load(open(task_json))
    def as_list(raw):
        if isinstance(raw, list):
            return raw
        try:
            return json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
    f2p, p2p = as_list(task.get("FAIL_TO_PASS")), as_list(task.get("PASS_TO_PASS"))

    stdout = open(stdout_f, errors="replace").read() if stdout_f else ""
    stderr = ""
    if stderr_f:
        try:
            stderr = open(stderr_f, errors="replace").read()
        except OSError:
            pass
    # runtests.py writes its progress to stderr, so both streams are candidates.
    passed, by_doc = parse_django(stdout + "\n" + stderr)

    verify = {
        "instance_id": task.get("instance_id"),
        "patch_source": patch_source,
        "model_patch_apply": model_apply,
        "test_patch_apply": test_apply,
    }

    if not passed and not by_doc:
        crash = next((s for s in CRASH_SIGNS if s in stderr), None)
        verify.update({
            "resolved": None, "status": "error",
            "error": (f"test process did not produce results: '{crash}'." if crash
                      else "runtests.py produced no parseable verbose output."),
            "f2p": {"passed": 0, "total": len(f2p), "pct": 0.0,
                    "tests": [{"test": t, "status": "not_run", "passed": False} for t in f2p]},
            "p2p": {"passed": 0, "total": len(p2p), "pct": 0.0,
                    "tests": [{"test": t, "status": "not_run", "passed": False} for t in p2p]},
        })
    else:
        f2p_g = group(f2p, check(f2p, passed, by_doc))
        p2p_g = group(p2p, check(p2p, passed, by_doc))
        resolved = (len(f2p) > 0 and f2p_g["passed"] == len(f2p)
                    and p2p_g["passed"] == len(p2p)
                    and model_apply == "OK" and test_apply == "OK")
        verify.update({"resolved": resolved,
                       "status": "resolved" if resolved else "not_resolved",
                       "result_source": "django-unittest",
                       "f2p": f2p_g, "p2p": p2p_g})

    json.dump(verify, open(out_json, "w"), indent=2)
    st = verify.get("status")
    if st == "error":
        print(f"[verify] {verify['instance_id']} [{patch_source}]: COULD NOT VERIFY")
        print(f"[verify]   {verify['error']}")
    else:
        r = "RESOLVED" if verify.get("resolved") else "NOT resolved"
        f, p = verify["f2p"], verify["p2p"]
        print(f"[verify] {verify['instance_id']} [{patch_source}]: {r}  (src=django-unittest)")
        print(f"[verify]   apply: model={model_apply} test={test_apply}")
        print(f"[verify]   F2P: {f['passed']}/{f['total']} ({f['pct']}%)   "
              f"P2P: {p['passed']}/{p['total']} ({p['pct']}%)")


if __name__ == "__main__":
    main()
