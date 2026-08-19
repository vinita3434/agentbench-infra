#!/usr/bin/env python3
"""Fetch the easiest SWE-bench Lite tasks into their own dataset.

Kept entirely separate from the SWE-PolyBench set in tasks/data/. Different
benchmark, different eval images, different verification path -- mixing them in
one directory would let a PolyBench-shaped verifier be pointed at a SWE-bench
task and fail in a way that looks like a model result.

SWE-bench Lite is already a curated easy subset (300 of the 2,294 full tasks:
single-file fixes, no new-feature work). This ranks within it by what actually
drives agent effort: how much code the fix touches, how many tests must flip,
and how much issue text has to be read first.

Output:
    tasks/swebench_lite/data/<instance_id>.json
    tasks/swebench_lite/manifest.jsonl

Usage:
    python tasks/fetch_swebench_lite.py --n 35
"""
import argparse
import json
import pathlib
import sys

DATASET = "princeton-nlp/SWE-bench_Lite"
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "swebench_lite"
DATA_DIR = OUT_DIR / "data"
MANIFEST = OUT_DIR / "manifest.jsonl"

# SWE-bench field names differ from PolyBench: FAIL_TO_PASS/PASS_TO_PASS are
# JSON-string lists, there is no test_command (the official harness derives it
# per repo), and environment_setup_commit pins the dependency install.
KEEP_FIELDS = [
    "instance_id", "repo", "base_commit", "problem_statement", "patch",
    "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit",
    "version", "created_at", "hints_text",
]


def listlen(raw):
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return len(v) if isinstance(v, (list, tuple)) else -1
    except (json.JSONDecodeError, TypeError):
        return -1


def ease_key(row):
    """Ascending: smaller is easier.

    Patch size first -- it is the strongest signal of how much has to be
    understood. Then F2P count, since every one is a separate condition the fix
    must satisfy. Problem-statement length last: long issues are noise to wade
    through, not extra work.
    """
    return (
        len(row.get("patch") or ""),
        listlen(row.get("FAIL_TO_PASS")),
        len(row.get("problem_statement") or ""),
        listlen(row.get("PASS_TO_PASS")),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=35)
    p.add_argument("--split", default="test")
    p.add_argument("--max-repo", type=int, default=8,
                   help="cap per repo so one project cannot dominate the set")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    print(f"[lite] loading {DATASET} split={args.split} ...", file=sys.stderr)
    rows = list(load_dataset(DATASET, split=args.split))
    print(f"[lite] {len(rows)} instances available", file=sys.stderr)

    rows.sort(key=ease_key)

    # Per-repo cap: SWE-bench Lite is heavily weighted toward django, and an
    # unconstrained easiest-N would be almost entirely one codebase -- which
    # measures familiarity with django, not coding ability.
    picked, per_repo = [], {}
    for row in rows:
        repo = row["repo"]
        if per_repo.get(repo, 0) >= args.max_repo:
            continue
        per_repo[repo] = per_repo.get(repo, 0) + 1
        picked.append(row)
        if len(picked) >= args.n:
            break

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w") as man:
        for row in picked:
            rec = {k: row.get(k) for k in KEEP_FIELDS}
            iid = rec["instance_id"]
            (DATA_DIR / f"{iid}.json").write_text(json.dumps(rec, indent=2))
            man.write(json.dumps({
                "instance_id": iid,
                "repo": rec["repo"],
                "language": "Python",          # SWE-bench Lite is Python-only
                "base_commit": rec["base_commit"],
                "environment_setup_commit": rec["environment_setup_commit"],
                "patch_len": len(rec["patch"] or ""),
                "n_f2p": listlen(rec["FAIL_TO_PASS"]),
                "n_p2p": listlen(rec["PASS_TO_PASS"]),
                "ps_len": len(rec["problem_statement"] or ""),
            }) + "\n")

    print(f"[lite] wrote {len(picked)} tasks -> {DATA_DIR}", file=sys.stderr)
    print(f"[lite] by repo: {per_repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
