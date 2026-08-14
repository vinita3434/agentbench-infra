#!/usr/bin/env python3
"""Fetch SWE-PolyBench tasks and prep them for the harness.

SWE-PolyBench is ~83% JS/TS, so a naive head() sample is useless for
per-language balance. This pulls a *stratified* subset: you say how many tasks
and which languages, and it draws evenly across the languages you ask for.

Output (consumed by run/run_task.sh):
    tasks/data/<instance_id>.json     one file per task
    tasks/manifest.jsonl              index: one line per selected task

Each per-task file stores the fields the harness needs to run and (later) to
verify the task:
    instance_id, repo, base_commit, problem_statement, test_patch, language
plus patch (gold), F2P, P2P, test_command  -- kept for the resolve-rate check.

Examples:
    # 10 tasks, evenly across the four languages
    python tasks/fetch_tasks.py --n 10

    # 12 Python + TS only, 6 each
    python tasks/fetch_tasks.py --n 12 --languages Python Typescript

    # deterministic subset for reproducibility
    python tasks/fetch_tasks.py --n 10 --seed 7

    # 50 easiest tasks, weighted 50% Python / 25% Java / 25% JS
    python tasks/fetch_tasks.py --n 50 --simplest \
        --mix Python=50 Java=25 JavaScript=25
"""
import argparse
import ast
import json
import pathlib
import random
import sys

import verifiability

DATASET = "AmazonScience/SWE-PolyBench"
# The dataset's `language` column uses these EXACT spellings (verified against
# the live split: capital-S JavaScript/TypeScript -- not "Javascript"). CLI
# input is normalized to these case-insensitively, so `typescript` also works.
ALL_LANGUAGES = ["Python", "Java", "JavaScript", "TypeScript"]
_CANONICAL = {lang.lower(): lang for lang in ALL_LANGUAGES}

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MANIFEST = HERE / "manifest.jsonl"

# Fields carried into each per-task file. Required-by-harness first, then the
# extras needed to actually score resolve rate later.
KEEP_FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
    "language",
    # ---- task classification + difficulty signals (for reports) ----
    "task_category",     # Bug Fix | Refactoring | Feature
    "num_nodes",         # code nodes the gold patch changes
    "num_func_changes",
    "num_class_changes",
    "is_single_func",
    "is_single_class",
    "is_mixed",
    # ---- kept for later verification, not needed to *run* a task ----
    "is_func_only",
    "is_class_only",
    "is_no_nodes",
    # ---- kept for later verification, not needed to *run* a task ----
    "patch",         # gold patch (minus tests)
    "F2P",           # fail-to-pass tests (JSON list as string)
    "P2P",           # pass-to-pass tests (JSON list as string)
    "test_command",  # how F2P/P2P were measured
]


def _listlen(s):
    """F2P/P2P are Python-repr strings ("['a', 'b']"), not JSON. -1 if unparseable."""
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else s
        return len(v) if isinstance(v, (list, tuple)) else -1
    except (ValueError, SyntaxError):
        return -1


def is_simple(row):
    """Hard gate for the 'simplest' pool.

    A task qualifies only if the gold patch touches exactly one function in one
    place and there is at least one F2P test to score against. Tasks with no
    F2P can never be marked resolved, so they are useless as benchmark items
    regardless of how small their diff is.
    """
    return (
        row.get("num_nodes") == 1
        and bool(row.get("is_single_func"))
        and not row.get("is_mixed")
        and 1 <= _listlen(row.get("F2P")) <= 5
        and len(row.get("patch") or "") <= 1500
    )


def simplicity_key(row):
    """Ascending sort key — smaller is simpler.

    Ordered by what most drives agent difficulty: how much code must change,
    then how many tests must be satisfied, then how much the agent has to read.
    P2P count is last; it costs verification wall time, not agent effort.
    """
    return (
        row.get("num_nodes") or 0,
        row.get("num_func_changes") or 0,
        _listlen(row.get("F2P")),
        len(row.get("patch") or ""),
        len(row.get("problem_statement") or ""),
        _listlen(row.get("P2P")),
    )


def parse_mix(pairs, languages, n):
    """'Python=50 Java=25' -> integer per-language quotas summing to exactly n.

    Weights are relative, so they need not sum to 100. Largest-remainder
    apportionment keeps the split as close to the requested ratio as integers
    allow while guaranteeing the quotas total n.
    """
    weights = {}
    for p in pairs:
        if "=" not in p:
            sys.exit(f"--mix expects LANG=WEIGHT, got {p!r}")
        lang, _, w = p.partition("=")
        canon = _CANONICAL.get(lang.lower())
        if not canon:
            sys.exit(f"unknown language in --mix: {lang!r}. Valid: {ALL_LANGUAGES}")
        if canon not in languages:
            sys.exit(f"--mix names {canon}, which is not in --languages {languages}")
        try:
            weights[canon] = float(w)
        except ValueError:
            sys.exit(f"--mix weight for {canon} is not a number: {w!r}")
    if not weights or sum(weights.values()) <= 0:
        sys.exit("--mix needs at least one positive weight")

    total = sum(weights.values())
    exact = {lang: n * w / total for lang, w in weights.items()}
    quotas = {lang: int(v) for lang, v in exact.items()}
    # Hand out the leftover seats to the largest fractional parts.
    for lang in sorted(exact, key=lambda k: exact[k] - quotas[k], reverse=True):
        if sum(quotas.values()) >= n:
            break
        quotas[lang] += 1
    return quotas


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10,
                   help="total number of tasks to select (default 10)")
    p.add_argument("--languages", nargs="+", default=ALL_LANGUAGES,
                   metavar="LANG",
                   help=f"languages to draw from (default: {' '.join(ALL_LANGUAGES)})")
    p.add_argument("--split", default="test",
                   help="dataset split to load (default: test)")
    p.add_argument("--seed", type=int, default=13,
                   help="RNG seed for reproducible sampling (default 13)")
    p.add_argument("--clean", action="store_true",
                   help="wipe tasks/data and manifest before writing")
    p.add_argument("--mix", nargs="+", metavar="LANG=WEIGHT", default=None,
                   help="relative per-language weights, e.g. Python=50 Java=25 "
                        "JavaScript=25 (default: even split across --languages)")
    p.add_argument("--simplest", action="store_true",
                   help="restrict to single-function, small-diff, 1-5 F2P tasks "
                        "and take the simplest ones instead of sampling randomly")
    p.add_argument("--verifiable", action="store_true",
                   help="keep only tasks the harness can score: repo's suite has "
                        "graded green (or matches one that has) AND the GHCR eval "
                        "image exists. Hits the registry; see tasks/verifiability.py")
    return p.parse_args()


def screen_verifiable(rows_by_lang, languages, cap, probe_chunk=32):
    """Keep only tasks the harness can actually score, cheapest test first.

    Two gates, in cost order: the repo's test suite must be gradeable at all
    (gold evidence, free, see tasks/verifiability.py), and the per-instance
    eval image must exist on GHCR (one registry call each). Images are probed
    simplest-first in chunks and the walk stops as soon as a language has `cap`
    survivors, so a 550-task pool costs a few hundred probes, not 550.
    """
    RUNNABLE, image_map, status = (verifiability.RUNNABLE,
                                   verifiability.image_map, verifiability.status)
    screened, dropped = {}, {}
    for lang in languages:
        pool = sorted(rows_by_lang.get(lang, []), key=simplicity_key)
        eligible, by_family = [], {}
        for row in pool:
            state, _ = status(row.get("repo"))
            if state in RUNNABLE:
                eligible.append(row)
            else:
                by_family[state] = by_family.get(state, 0) + 1
        dropped[lang] = by_family

        keep, i = [], 0
        while i < len(eligible) and len(keep) < cap:
            chunk = eligible[i:i + probe_chunk]
            have = image_map([r["instance_id"] for r in chunk])
            keep.extend(r for r in chunk if have[r["instance_id"]])
            i += probe_chunk
        screened[lang] = keep[:cap]
        print(f"[fetch] {lang}: {len(pool)} simple -> {len(eligible)} gradeable "
              f"-> {len(screened[lang])} with eval image "
              f"(probed {min(i, len(eligible))})", file=sys.stderr)

    for lang, counts in dropped.items():
        if counts:
            print(f"[fetch] {lang} dropped as ungradeable: {counts}", file=sys.stderr)
    return screened


def stratified_pick(rows_by_lang, languages, n, rng, quotas=None, simplest=False):
    """Draw across `languages` per `quotas`, redistributing any shortfall.

    If a language has fewer tasks than its quota, the remainder is spread over
    the languages that still have supply, so we hit `n` whenever possible.
    With `simplest`, pools are ordered by `simplicity_key` instead of shuffled,
    which makes selection fully deterministic and ignores the seed.
    """
    if quotas is None:
        quotas = {}
        base, extra = divmod(n, len(languages))
        for i, lang in enumerate(languages):
            quotas[lang] = base + (1 if i < extra else 0)

    # Order each language's pool up front: simplest-first, or shuffled (via rng).
    pools = {}
    for lang in languages:
        pool = list(rows_by_lang.get(lang, []))
        if simplest:
            pool.sort(key=simplicity_key)
        else:
            rng.shuffle(pool)
        pools[lang] = pool

    selected = []
    # First pass: take up to quota from each language.
    for lang in languages:
        take = min(quotas.get(lang, 0), len(pools[lang]))
        selected.extend(pools[lang][:take])
        pools[lang] = pools[lang][take:]

    # Second pass: fill any shortfall from whatever pools still have supply.
    shortfall = n - len(selected)
    if shortfall > 0:
        leftovers = [r for lang in languages for r in pools[lang]]
        if simplest:
            leftovers.sort(key=simplicity_key)
        else:
            rng.shuffle(leftovers)
        selected.extend(leftovers[:shortfall])

    return selected[:n]


def main():
    args = parse_args()

    # Normalize CLI languages to canonical spellings (case-insensitive).
    normalized, bad = [], []
    for l in args.languages:
        canon = _CANONICAL.get(l.lower())
        (normalized if canon else bad).append(canon or l)
    if bad:
        sys.exit(f"unknown language(s): {bad}. Valid: {ALL_LANGUAGES}")
    args.languages = normalized

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("The `datasets` library is required: pip install datasets")

    print(f"[fetch] loading {DATASET} split={args.split} ...", file=sys.stderr)
    ds = load_dataset(DATASET, split=args.split)

    # Bucket rows by language.
    rows_by_lang = {lang: [] for lang in ALL_LANGUAGES}
    for row in ds:
        lang = row.get("language")
        if lang in rows_by_lang:
            rows_by_lang[lang].append(row)
    avail = {lang: len(rows_by_lang[lang]) for lang in ALL_LANGUAGES}
    print(f"[fetch] available by language: {avail}", file=sys.stderr)

    if args.simplest:
        rows_by_lang = {lang: [r for r in rows if is_simple(r)]
                        for lang, rows in rows_by_lang.items()}
        pool_sizes = {lang: len(rows_by_lang[lang]) for lang in ALL_LANGUAGES}
        print(f"[fetch] simple-pool by language: {pool_sizes}", file=sys.stderr)

    if args.verifiable:
        # Cap per language at n: a language can at most fill the whole request
        # once shortfall from the others is redistributed to it.
        rows_by_lang = screen_verifiable(rows_by_lang, args.languages, args.n)

    # --mix defines the quotas *and* narrows the language set to what it names.
    quotas = None
    if args.mix:
        quotas = parse_mix(args.mix, args.languages, args.n)
        args.languages = [l for l in args.languages if l in quotas]
        print(f"[fetch] quotas: {quotas}", file=sys.stderr)
        for lang, q in quotas.items():
            have = len(rows_by_lang.get(lang, []))
            if have < q:
                print(f"[fetch] WARNING: {lang} quota {q} exceeds pool {have}; "
                      f"shortfall will be filled from other languages",
                      file=sys.stderr)

    rng = random.Random(args.seed)
    picked = stratified_pick(rows_by_lang, args.languages, args.n, rng,
                             quotas=quotas, simplest=args.simplest)
    if not picked:
        sys.exit("[fetch] no tasks selected -- check --languages/--n/--simplest")
    if len(picked) < args.n:
        print(f"[fetch] WARNING: asked for {args.n}, only {len(picked)} available",
              file=sys.stderr)

    if args.clean and DATA_DIR.exists():
        for f in DATA_DIR.glob("*.json"):
            f.unlink()
        MANIFEST.unlink(missing_ok=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    got_by_lang = {}
    with MANIFEST.open("w") as man:
        for row in picked:
            record = {k: row.get(k) for k in KEEP_FIELDS}
            iid = record["instance_id"]
            (DATA_DIR / f"{iid}.json").write_text(json.dumps(record, indent=2))
            man.write(json.dumps({
                "instance_id": iid,
                "repo": record["repo"],
                "language": record["language"],
                "base_commit": record["base_commit"],
                "task_category": record["task_category"],
                "num_nodes": record["num_nodes"],
                "n_f2p": _listlen(record["F2P"]),
                "n_p2p": _listlen(record["P2P"]),
                "patch_len": len(record["patch"] or ""),
                # Why we believe this task is scoreable, carried through so a
                # report can separate "model failed" from "we never proved the
                # verifier works here".
                "verify_status": verifiability.status(record["repo"])[0],
                "test_framework": verifiability.framework(record.get("test_command")),
            }) + "\n")
            got_by_lang[record["language"]] = got_by_lang.get(record["language"], 0) + 1

    print(f"[fetch] wrote {len(picked)} tasks to {DATA_DIR}", file=sys.stderr)
    print(f"[fetch] selected by language: {got_by_lang}", file=sys.stderr)
    print(f"[fetch] manifest -> {MANIFEST}", file=sys.stderr)


if __name__ == "__main__":
    main()
