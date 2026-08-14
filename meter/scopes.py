#!/usr/bin/env python3
"""Two measurement scopes, kept apart by construction.

This module exists because of one specific bug: a prefix cache *rate* above 1.
It happens when a server-wide cumulative counter is used as the numerator over a
single request's prompt tokens. The two numbers look interchangeable -- both are
"cached tokens" -- but they answer different questions over different windows,
and dividing one by the other is meaningless.

So the scopes are separate types, not separate fields on one type:

    RequestUsage    one request. Numbers describe THAT completion only.
                    Source: the `usage` object on the completion response.

    ServerSnapshot  the whole server, at an instant. Counters are cumulative
                    since process start; gauges are point-in-time.
                    Source: SGLang's Prometheus /metrics endpoint.

Anything computing a per-request ratio accepts `RequestUsage` and nothing else.
A `ServerSnapshot` cannot be passed where a `RequestUsage` is required, so the
bug cannot be reintroduced by accident -- it would be a type error, and the
guards in ratios.py catch it if someone unpacks the values by hand anyway.

Missing stays missing: every field is Optional and defaults to None. A field the
server did not report is None, never 0. Downstream, "absent" and "zero" are
different facts and must stay distinguishable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# --- per-request ---------------------------------------------------------


@dataclass(frozen=True)
class RequestUsage:
    """Token accounting for exactly one completion.

    `cached_tokens` is the subset of `prompt_tokens` the server served from its
    prefix cache instead of recomputing. It is bounded by prompt_tokens by
    definition -- if it is not, the value did not come from this request.
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    # Kept so a later reader can see exactly what the server said, including
    # fields this code does not interpret.
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def scope(self) -> str:
        return "request"


def _as_int(value: Any) -> Optional[int]:
    """int(value) or None. Never coerces a missing/garbage value into 0."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def usage_from_response(body: Dict[str, Any]) -> RequestUsage:
    """Extract per-request usage from an OpenAI-compatible completion body.

    Servers disagree on where cached prompt tokens live. OpenAI and SGLang put
    it at usage.prompt_tokens_details.cached_tokens; some builds surface a flat
    usage.cached_tokens. Both are per-request, so either is a valid source here.
    A server that reports neither leaves the field None -- not 0, because "this
    server does not report cache hits" is not the same as "nothing was cached".
    """
    usage = (body or {}).get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}

    cached = _as_int(details.get("cached_tokens"))
    if cached is None:
        cached = _as_int(usage.get("cached_tokens"))

    return RequestUsage(
        prompt_tokens=_as_int(usage.get("prompt_tokens")),
        completion_tokens=_as_int(usage.get("completion_tokens")),
        total_tokens=_as_int(usage.get("total_tokens")),
        cached_tokens=cached,
        raw=dict(usage),
    )


# --- server-wide ---------------------------------------------------------


@dataclass(frozen=True)
class ServerSnapshot:
    """Whole-server state at one instant, scraped from Prometheus.

    Gauges (kv pool utilization) are meaningful on their own: they describe the
    server *now*. Counters (cumulative cache hits since boot) are meaningful
    only as differences between two snapshots, and are NEVER a per-request
    numerator -- that is the bug this module exists to prevent.
    """

    scraped_at: Optional[float] = None
    kv_pool_utilization: Optional[float] = None       # gauge, 0..1
    cache_hit_rate_serverwide: Optional[float] = None  # gauge, 0..1, server's own
    running_requests: Optional[float] = None           # gauge
    queued_requests: Optional[float] = None            # gauge
    gen_throughput_serverwide: Optional[float] = None  # gauge, tok/s
    raw: Dict[str, float] = field(default_factory=dict)

    @property
    def scope(self) -> str:
        return "server"


# SGLang has renamed these across versions, so each metric lists the names seen
# in the wild, most current first. Unknown builds simply yield None.
_GAUGE_CANDIDATES = {
    "kv_pool_utilization": (
        "sglang:token_usage",
        "sglang:kv_cache_usage",
        "sglang:token_usage_ratio",
    ),
    "cache_hit_rate_serverwide": (
        "sglang:cache_hit_rate",
        "sglang:tree_cache_hit_rate",
    ),
    "running_requests": ("sglang:num_running_reqs", "sglang:running_requests"),
    "queued_requests": ("sglang:num_queue_reqs", "sglang:queued_requests"),
    "gen_throughput_serverwide": ("sglang:gen_throughput",),
}

_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)\s*$"
)


def parse_prometheus(text: str) -> Dict[str, float]:
    """Flatten a Prometheus exposition body to {metric_name: value}.

    Labels are dropped: a single-model server exposes one series per metric, and
    keeping labels here would invite aggregating across series, which is exactly
    the kind of scope mixing this module forbids. Non-numeric values (NaN, Inf)
    are skipped rather than stored -- they are not measurements.
    """
    out: Dict[str, float] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        out.setdefault(m.group("name"), value)
    return out


def snapshot_from_prometheus(text: str, scraped_at: Optional[float] = None) -> ServerSnapshot:
    """Build a ServerSnapshot from a scrape. Absent metrics stay None."""
    flat = parse_prometheus(text)
    picked: Dict[str, Optional[float]] = {}
    for field_name, candidates in _GAUGE_CANDIDATES.items():
        picked[field_name] = next((flat[c] for c in candidates if c in flat), None)
    return ServerSnapshot(scraped_at=scraped_at, raw=flat, **picked)
