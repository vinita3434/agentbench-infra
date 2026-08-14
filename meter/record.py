#!/usr/bin/env python3
"""Build one raw row per LLM call, and append it to disk.

Two rules shape this module.

**Raw rows only.** One record per call, no aggregation, no running means, no
derived-from-other-rows fields. Aggregation is a separate offline step over the
JSONL. Collecting and summarizing in the same pass makes it impossible to
recompute a summary after finding a bug in it, and there is always a bug in it.

**Metrics never break an episode.** Every computation is wrapped: a failure
records the exception on the row and leaves the field absent, then the call
returns normally. Losing a trajectory costs far more than losing a metric, so
nothing in this path is allowed to raise into the request handler.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

from .perf import Timing, compute_perf
from .prefix import compute_prefix_metrics
from .ratios import Ratio
from .scopes import RequestUsage, ServerSnapshot

log = logging.getLogger("meter.record")

ROWS_FILENAME = "turns.jsonl"
ERRORS_FILENAME = "meter_errors.log"


def _safe(field_name: str, fn, errors: List[Dict[str, str]], default=None):
    """Run a metric computation; on failure record it and carry on.

    The error is appended to the row rather than only logged, so a row with a
    missing metric always carries the reason it is missing. Silent gaps are
    indistinguishable from metrics that were never configured.
    """
    try:
        return fn()
    except Exception as exc:  # deliberately broad: no metric may kill an episode
        log.error("metric %s failed: %s", field_name, exc)
        errors.append({
            "field": field_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        })
        return default


def build_row(
    *,
    call_index: int,
    usage: RequestUsage,
    timing: Timing,
    previous_messages: Optional[Sequence[Any]] = None,
    current_messages: Optional[Sequence[Any]] = None,
    server_before: Optional[ServerSnapshot] = None,
    server_after: Optional[ServerSnapshot] = None,
    token_counter=None,
    prefix_counter=None,
    active_param_count: Optional[float] = None,
    bytes_per_param: Optional[float] = None,
    peak_bandwidth_bytes_per_s: Optional[float] = None,
    run_label: Optional[str] = None,
    task_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One flat record for one LLM call. Never raises."""
    errors: List[Dict[str, str]] = []

    row: Dict[str, Any] = {
        "schema": "meter.turn.v1",
        "recorded_at_unix": time.time(),
        "run_label": run_label,
        "task_id": task_id,
        # Turn index within the episode. The whole point of the exercise is how
        # these metrics move as context accumulates, so ordering is data.
        "call_index": call_index,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cached_tokens": usage.cached_tokens,
    }

    prefix = _safe(
        "prefix",
        lambda: compute_prefix_metrics(
            usage, previous_messages, current_messages, token_counter, prefix_counter
        ),
        errors,
    )
    if prefix is not None:
        row.update(prefix.as_dict())

    perf = _safe(
        "perf",
        lambda: compute_perf(
            usage.completion_tokens,
            timing,
            param_count=active_param_count,
            bytes_per_param=bytes_per_param,
            peak_bandwidth_bytes_per_s=peak_bandwidth_bytes_per_s,
        ),
        errors,
    )
    if perf is not None:
        row.update(perf.as_dict())

    # Server-wide gauges stay explicitly namespaced. They are NOT per-request,
    # and the names say so, so nobody divides one by a per-request token count.
    for label, snap in (("server_before", server_before), ("server_after", server_after)):
        if snap is None:
            row[f"{label}_kv_pool_utilization"] = None
            row[f"{label}_running_requests"] = None
            row[f"{label}_queued_requests"] = None
            row[f"{label}_scraped_at"] = None
            continue
        row[f"{label}_kv_pool_utilization"] = snap.kv_pool_utilization
        row[f"{label}_running_requests"] = snap.running_requests
        row[f"{label}_queued_requests"] = snap.queued_requests
        row[f"{label}_scraped_at"] = snap.scraped_at
        # Recorded for comparison against our per-request rate -- never as a
        # substitute for it. Different scope, different question.
        row[f"{label}_cache_hit_rate_serverwide"] = snap.cache_hit_rate_serverwide

    if extra:
        row.update(extra)

    row["meter_errors"] = errors or None
    return row


class RowWriter:
    """Append-only JSONL writer, one file per episode.

    Flushes every row: an episode that dies mid-trajectory must still leave
    every completed turn on disk. Buffering would trade exactly the data we
    cannot afford to lose for throughput we do not need -- these are a few
    hundred rows per episode, not a hot path.
    """

    def __init__(self, directory: os.PathLike, filename: str = ROWS_FILENAME):
        self.dir = pathlib.Path(directory)
        self.path = self.dir / filename
        self.error_path = self.dir / ERRORS_FILENAME
        self._lock = threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, row: Dict[str, Any]) -> None:
        """Append one row. Failure here is logged, never raised."""
        try:
            line = json.dumps(row, ensure_ascii=False, default=str)
        except Exception as exc:
            self.log_error(f"row serialization failed: {exc}")
            return
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        except Exception as exc:
            self.log_error(f"row write failed: {exc}")

    def log_error(self, message: str) -> None:
        """Sidecar error log: metering problems must be visible without being fatal."""
        log.error("%s", message)
        try:
            with self._lock:
                with open(self.error_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
        except Exception:
            pass  # nothing left to do; the episode still matters more
