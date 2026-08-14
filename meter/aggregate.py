#!/usr/bin/env python3
"""Offline summaries over turns.jsonl. Never runs during collection.

Collection emits raw rows only, so every summary here is recomputable after the
fact. That matters because summaries acquire bugs, and a summary you cannot
recompute is a summary you cannot fix.

The headline number is a **pooled** ratio, not a mean of ratios:

    weighted_prefix_efficiency = Σ cached_within_episode / Σ reusable_prefix_tokens

That is algebraically identical to weighting each turn's efficiency by its
reusable tokens:

    Σ (efficiency_n × reusable_n)   Σ (cached_within_n/reusable_n × reusable_n)
    ────────────────────────────  = ─────────────────────────────────────────── = Σ within / Σ reusable
          Σ reusable_n                          Σ reusable_n

so it needs no per-turn division and no weight normalisation to get wrong. It
also handles turn 1 for free: `reusable = 0` contributes 0 to both sums, so a
row with nothing to hit cannot drag the average down.

Weighting by reusable tokens is the point. A miss on turn 40 with 40k eligible
tokens is a far bigger loss than a miss on turn 2 with 1.2k, and a plain mean
would call them equal.

Usage:
    python -m meter.aggregate results/serving/qwen3-coder-30b/<task>/turns.jsonl
    python -m meter.aggregate --sweep results/serving/qwen3-coder-30b --csv out.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional

SUMMARY_FILENAME = "episode_summary.json"


def read_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load a turns.jsonl. A truncated final line is tolerated and counted.

    An episode killed mid-write still leaves usable rows, and refusing to read
    them would lose exactly the trajectory we said matters more than a metric.
    """
    rows, bad = [], 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[aggregate] {path}: skipped {bad} unparseable line(s)", file=sys.stderr)
    return rows


def _defined(values: Iterable[Optional[float]]) -> List[float]:
    """Drop None. Never substitutes 0 -- absent and zero are different facts."""
    return [v for v in values if v is not None]


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Episode-level summary. Every aggregate reports the n it came from."""
    out: Dict[str, Any] = {
        "schema": "meter.episode_summary.v1",
        "turns": len(rows),
    }
    if not rows:
        return out

    out["run_label"] = rows[0].get("run_label")
    out["task_id"] = rows[0].get("task_id")

    # --- prefix reuse, pooled ------------------------------------------
    # Only rows where the denominator was actually measured. A row with
    # reusable=None was never counted; treating it as 0 would quietly enlarge
    # the numerator's share and inflate efficiency.
    usable = [
        r for r in rows
        if r.get("prefix_reusable_tokens") is not None
        and r.get("cached_within_episode") is not None
    ]
    sum_reusable = sum(r["prefix_reusable_tokens"] for r in usable)
    sum_within = sum(r["cached_within_episode"] for r in usable)

    out["turns_with_measured_reusable"] = len(usable)
    out["sum_reusable_tokens"] = sum_reusable
    out["sum_cached_within_episode"] = sum_within

    if sum_reusable > 0:
        out["weighted_prefix_efficiency"] = sum_within / sum_reusable
        out["weighted_prefix_efficiency_reason"] = None
        # The absolute counterpart: how many tokens had to be recomputed even
        # though they were eligible. A ratio says how well; this says how much,
        # and converts directly to GPU-seconds of wasted prefill.
        out["recomputed_tokens"] = sum_reusable - sum_within
    else:
        # Every row had nothing eligible (single-turn episode, or no tokenizer).
        out["weighted_prefix_efficiency"] = None
        out["weighted_prefix_efficiency_reason"] = "undefined:zero_denominator"
        out["recomputed_tokens"] = None

    # Unweighted mean, for contrast only. If these differ a lot, the losses are
    # concentrated in the turns that carried the most tokens -- which is the
    # thing the weighting exists to surface.
    per_turn = _defined(r.get("weighted_prefix_efficiency") for r in rows)
    out["unweighted_mean_efficiency"] = (
        sum(per_turn) / len(per_turn) if per_turn else None
    )
    out["turns_with_defined_efficiency"] = len(per_turn)

    # --- cache rate, pooled (work avoided) -----------------------------
    prompt_rows = [
        r for r in rows
        if r.get("prompt_tokens") is not None and r.get("cached_tokens") is not None
    ]
    sum_prompt = sum(r["prompt_tokens"] for r in prompt_rows)
    sum_cached = sum(r["cached_tokens"] for r in prompt_rows)
    out["sum_prompt_tokens"] = sum_prompt
    out["sum_cached_tokens"] = sum_cached
    out["prefix_cache_rate_pooled"] = (sum_cached / sum_prompt) if sum_prompt else None
    out["turns_with_token_counts"] = len(prompt_rows)

    # Cross-episode carryover: prefix the server held from earlier tasks.
    beyond = _defined(r.get("cached_beyond_episode") for r in rows)
    out["cached_beyond_episode_total"] = sum(beyond) if beyond else None
    out["turns_spanning_episodes"] = sum(
        1 for r in rows if r.get("prefix_reuse_scope") == "spans_episodes"
    )

    # --- throughput and bandwidth --------------------------------------
    decode = _defined(r.get("decode_tokens_per_s") for r in rows)
    out["decode_tps_median"] = _median(decode)
    out["decode_tps_min"] = min(decode) if decode else None
    out["decode_tps_max"] = max(decode) if decode else None
    out["turns_with_decode_rate"] = len(decode)

    mbu = _defined(r.get("mbu") for r in rows)
    out["mbu_median"] = _median(mbu)
    out["turns_with_mbu"] = len(mbu)

    ttft = _defined(r.get("ttft_s") for r in rows)
    out["ttft_median_s"] = _median(ttft)
    # TTFT drift is the context-growth signal: first turn vs last turn.
    out["ttft_first_s"] = rows[0].get("ttft_s")
    out["ttft_last_s"] = rows[-1].get("ttft_s")

    # --- server pressure ------------------------------------------------
    kv = _defined(r.get("server_after_kv_pool_utilization") for r in rows)
    out["kv_pool_utilization_max"] = max(kv) if kv else None
    out["kv_pool_utilization_mean"] = (sum(kv) / len(kv)) if kv else None
    out["turns_with_kv_gauge"] = len(kv)

    # --- context growth --------------------------------------------------
    prompts = _defined(r.get("prompt_tokens") for r in rows)
    out["prompt_tokens_first"] = prompts[0] if prompts else None
    out["prompt_tokens_last"] = prompts[-1] if prompts else None
    out["prompt_tokens_max"] = max(prompts) if prompts else None

    # --- why anything was missing ----------------------------------------
    # Reasons are counted, not summarised away: "efficiency absent on 40 of 60
    # rows because no tokenizer" is a different situation from "absent on 1".
    reasons: collections.Counter = collections.Counter()
    for r in rows:
        for key in (
            "weighted_prefix_efficiency_reason",
            "prefix_cache_rate_reason",
            "decode_tokens_per_s_reason",
            "mbu_reason",
        ):
            if r.get(key):
                reasons[f"{key}={r[key]}"] += 1
    out["excluded_rows"] = dict(reasons) or None
    out["rows_with_meter_errors"] = sum(1 for r in rows if r.get("meter_errors"))
    return out


def summarize_file(path: pathlib.Path, write: bool = True) -> Dict[str, Any]:
    rows = read_rows(path)
    summary = summarize(rows)
    summary["source"] = str(path)
    if write:
        (path.parent / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    return summary


def summarize_sweep(root: pathlib.Path, write: bool = True) -> List[Dict[str, Any]]:
    """Every episode under a run label."""
    out = []
    for turns in sorted(root.rglob("turns.jsonl")):
        out.append(summarize_file(turns, write=write))
    return out


def _print(summary: Dict[str, Any]) -> None:
    def fmt(value, digits=3):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return f"{value:,}" if isinstance(value, int) else str(value)

    print(f"\n{summary.get('task_id')}  [{summary.get('run_label')}]  "
          f"{summary.get('turns')} turns")
    print(f"  weighted prefix efficiency  {fmt(summary.get('weighted_prefix_efficiency'))}"
          f"   (Σ{fmt(summary.get('sum_cached_within_episode'))}"
          f" / Σ{fmt(summary.get('sum_reusable_tokens'))}"
          f", n={summary.get('turns_with_measured_reusable')})")
    print(f"  unweighted mean             {fmt(summary.get('unweighted_mean_efficiency'))}"
          f"   (n={summary.get('turns_with_defined_efficiency')})")
    print(f"  recomputed tokens           {fmt(summary.get('recomputed_tokens'))}")
    print(f"  prefix cache rate (pooled)  {fmt(summary.get('prefix_cache_rate_pooled'))}")
    print(f"  cached beyond episode       {fmt(summary.get('cached_beyond_episode_total'))}"
          f"   ({summary.get('turns_spanning_episodes')} turns)")
    print(f"  decode tok/s (median)       {fmt(summary.get('decode_tps_median'), 1)}"
          f"   (n={summary.get('turns_with_decode_rate')})")
    print(f"  MBU (median)                {fmt(summary.get('mbu_median'))}"
          f"   (n={summary.get('turns_with_mbu')})")
    print(f"  KV pool max / mean          {fmt(summary.get('kv_pool_utilization_max'))}"
          f" / {fmt(summary.get('kv_pool_utilization_mean'))}")
    print(f"  TTFT first → last           {fmt(summary.get('ttft_first_s'), 2)}s → "
          f"{fmt(summary.get('ttft_last_s'), 2)}s")
    print(f"  prompt tokens first → last  {fmt(summary.get('prompt_tokens_first'))} → "
          f"{fmt(summary.get('prompt_tokens_last'))}")
    if summary.get("excluded_rows"):
        print(f"  missing:  {summary['excluded_rows']}")


CSV_FIELDS = [
    "run_label", "task_id", "turns",
    "weighted_prefix_efficiency", "unweighted_mean_efficiency", "recomputed_tokens",
    "prefix_cache_rate_pooled", "cached_beyond_episode_total",
    "decode_tps_median", "mbu_median",
    "kv_pool_utilization_max", "ttft_first_s", "ttft_last_s",
    "prompt_tokens_first", "prompt_tokens_last",
    "turns_with_measured_reusable", "turns_with_decode_rate", "turns_with_mbu",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="a turns.jsonl, or a run-label directory with --sweep")
    p.add_argument("--sweep", action="store_true",
                   help="summarize every episode under a run-label directory")
    p.add_argument("--csv", help="also write one row per episode to this CSV")
    p.add_argument("--no-write", action="store_true",
                   help="print only; do not write episode_summary.json")
    args = p.parse_args(argv)

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"[aggregate] no such path: {path}", file=sys.stderr)
        return 1

    summaries = (summarize_sweep(path, write=not args.no_write) if args.sweep
                 else [summarize_file(path, write=not args.no_write)])

    for s in summaries:
        _print(s)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)
        print(f"\n[aggregate] csv -> {args.csv}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
