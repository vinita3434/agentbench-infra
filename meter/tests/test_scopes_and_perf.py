#!/usr/bin/env python3
"""Scope parsing, decode-throughput windows, MBU bounds, and row integrity."""
import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from meter.perf import Timing, compute_perf, decode_tokens_per_s, mbu  # noqa: E402
from meter.record import RowWriter, build_row  # noqa: E402
from meter.scopes import (  # noqa: E402
    RequestUsage,
    parse_prometheus,
    snapshot_from_prometheus,
    usage_from_response,
)

# --- scopes ---------------------------------------------------------------


def test_usage_nested_cached_tokens():
    usage = usage_from_response({"usage": {
        "prompt_tokens": 1200, "completion_tokens": 64, "total_tokens": 1264,
        "prompt_tokens_details": {"cached_tokens": 1024},
    }})
    assert (usage.prompt_tokens, usage.cached_tokens) == (1200, 1024)
    assert usage.scope == "request"


def test_usage_flat_cached_tokens_fallback():
    usage = usage_from_response({"usage": {"prompt_tokens": 10, "cached_tokens": 4}})
    assert usage.cached_tokens == 4


def test_absent_usage_fields_stay_none():
    usage = usage_from_response({})
    assert usage.prompt_tokens is None and usage.cached_tokens is None


def test_prometheus_parse_skips_comments_and_nan():
    text = """# HELP sglang:token_usage KV pool
# TYPE sglang:token_usage gauge
sglang:token_usage 0.42
sglang:num_running_reqs 3
sglang:broken NaN
"""
    flat = parse_prometheus(text)
    assert flat["sglang:token_usage"] == pytest.approx(0.42)
    assert "sglang:broken" not in flat


def test_snapshot_scope_and_absent_metrics():
    snap = snapshot_from_prometheus("sglang:token_usage 0.61\n")
    assert snap.scope == "server"
    assert snap.kv_pool_utilization == pytest.approx(0.61)
    assert snap.cache_hit_rate_serverwide is None  # absent, not 0


# --- decode throughput ----------------------------------------------------


def test_decode_excludes_prefill():
    """Decode rate uses the first-token-to-last window, not total time."""
    t = Timing(started_at=100.0, first_token_at=102.0, finished_at=104.0)
    tps, reason = decode_tokens_per_s(101, t)
    assert reason is None
    assert tps == pytest.approx(50.0)      # (101-1)/2s, prefill's 2s excluded
    # Total-time throughput would be 101/4 = 25.25 -- half, and drifting with
    # prompt length. That is the number this metric exists to avoid.


def test_non_streamed_has_no_decode_rate():
    t = Timing(started_at=100.0, first_token_at=None, finished_at=104.0)
    tps, reason = decode_tokens_per_s(500, t)
    assert tps is None and reason == "undefined:not_streamed"


def test_single_token_completion_has_no_rate():
    t = Timing(started_at=1.0, first_token_at=2.0, finished_at=2.0)
    tps, reason = decode_tokens_per_s(1, t)
    assert tps is None and reason == "undefined:too_few_output_tokens"


# --- MBU ------------------------------------------------------------------


def test_mbu_typical_h100():
    """3.3B active params at fp8, 100 tok/s on an H100 SXM."""
    r = mbu(decode_tps=100.0, bytes_per_token=3.3e9, peak_bandwidth_bytes_per_s=3.35e12)
    assert r.value == pytest.approx(0.0985, abs=1e-3)


def test_mbu_above_one_is_refused():
    """Total instead of active params inflates MBU past 1 -- a wrong constant."""
    r = mbu(decode_tps=200.0, bytes_per_token=30.5e9, peak_bandwidth_bytes_per_s=3.35e12)
    assert r.value is None and r.reason == "invariant:ratio_above_one"


def test_mbu_missing_config_is_absent():
    assert mbu(100.0, None, 3.35e12).value is None
    assert mbu(100.0, 3.3e9, None).value is None


def test_compute_perf_shape():
    t = Timing(started_at=0.0, first_token_at=1.0, finished_at=3.0)
    row = compute_perf(201, t, 3.3e9, 1.0, 3.35e12).as_dict()
    assert row["decode_tokens_per_s"] == pytest.approx(100.0)
    assert row["ttft_s"] == pytest.approx(1.0)
    assert row["mbu"] is not None


# --- rows -----------------------------------------------------------------


def test_build_row_never_raises_on_bad_input():
    """A metric failure must leave a usable row, not kill the episode."""
    row = build_row(
        call_index=1,
        usage=RequestUsage(prompt_tokens=100, completion_tokens=10, cached_tokens=50),
        timing=Timing(started_at=0.0, first_token_at=1.0, finished_at=2.0),
        previous_messages=[{"role": "user", "content": "x"}],
        current_messages=[{"role": "user", "content": "x"}],
        token_counter=lambda t: 1 / 0,   # explodes on call
        run_label="qwen3-coder-30b",
        task_id="t1",
    )
    assert row["prompt_tokens"] == 100
    assert row["call_index"] == 1
    # reusable could not be counted, so efficiency is absent -- not fabricated.
    assert row["weighted_prefix_efficiency"] is None


def test_row_writer_appends_and_is_readable():
    with tempfile.TemporaryDirectory() as tmp:
        w = RowWriter(tmp)
        w.write({"call_index": 1, "prompt_tokens": 10})
        w.write({"call_index": 2, "prompt_tokens": 20})
        rows = [json.loads(l) for l in w.path.read_text().splitlines()]
        assert [r["call_index"] for r in rows] == [1, 2]


def test_row_writer_survives_unserializable_row():
    with tempfile.TemporaryDirectory() as tmp:
        w = RowWriter(tmp)
        w.write({"bad": object()})       # default=str handles it, must not raise
        w.write({"call_index": 1})
        assert w.path.exists()
