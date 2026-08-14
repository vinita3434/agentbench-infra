#!/usr/bin/env python3
"""Guards on ratios: the >1 regression, and absent-vs-zero."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from meter.ratios import (  # noqa: E402
    INVARIANT_ABOVE_ONE,
    INVARIANT_NEGATIVE,
    UNDEFINED_MISSING_INPUT,
    UNDEFINED_ZERO_DENOMINATOR,
    ScopeMixingError,
    bounded_ratio,
    require_request_scope,
)
from meter.scopes import ServerSnapshot, usage_from_response  # noqa: E402


def test_normal_ratio():
    r = bounded_ratio(300, 1000, name="t")
    assert r.value == pytest.approx(0.3)
    assert r.reason is None and r.ok


def test_ratio_of_one_is_valid():
    """A fully cached prompt is legitimate, not an invariant violation."""
    r = bounded_ratio(1000, 1000, name="t")
    assert r.value == pytest.approx(1.0) and r.ok


# --- the regression this module exists for -------------------------------


def test_ratio_above_one_yields_no_value():
    """REGRESSION: a rate above 1 must never reach the data.

    Reproduces the original bug: a server-wide cumulative cache counter
    (millions of tokens since boot) divided by one request's prompt tokens.
    """
    serverwide_cumulative_hits = 4_812_355
    this_request_prompt_tokens = 12_004

    r = bounded_ratio(
        serverwide_cumulative_hits, this_request_prompt_tokens, name="prefix_cache_rate"
    )

    assert r.value is None, "a >1 ratio must not be emitted as a number"
    assert r.reason == INVARIANT_ABOVE_ONE
    # The inputs survive so the scope error is diagnosable from the row alone.
    assert r.numerator == serverwide_cumulative_hits
    assert r.denominator == this_request_prompt_tokens


def test_scope_guard_rejects_server_snapshot():
    """Handing a server-wide object to a per-request computation fails loudly."""
    snapshot = ServerSnapshot(cache_hit_rate_serverwide=0.93)
    with pytest.raises(ScopeMixingError):
        require_request_scope(snapshot, name="prefix_cache_rate")


def test_scope_guard_accepts_request_usage():
    usage = usage_from_response({"usage": {"prompt_tokens": 10}})
    require_request_scope(usage, name="prefix_cache_rate")  # must not raise


# --- absent vs zero -------------------------------------------------------


def test_zero_denominator_is_undefined_not_zero():
    """Nothing reusable is not a 0% hit rate; it is no rate at all."""
    r = bounded_ratio(0, 0, name="t")
    assert r.value is None
    assert r.reason == UNDEFINED_ZERO_DENOMINATOR
    assert r.value != 0.0


def test_missing_input_is_undefined():
    assert bounded_ratio(None, 100, name="t").reason == UNDEFINED_MISSING_INPUT
    assert bounded_ratio(50, None, name="t").reason == UNDEFINED_MISSING_INPUT


def test_genuine_zero_is_preserved():
    """A real 0 -- nothing cached out of a reusable prefix -- stays 0."""
    r = bounded_ratio(0, 500, name="t")
    assert r.value == 0.0 and r.ok and r.reason is None


def test_negative_input_rejected():
    r = bounded_ratio(-5, 100, name="t")
    assert r.value is None and r.reason == INVARIANT_NEGATIVE


def test_as_dict_carries_reason_and_inputs():
    row = bounded_ratio(9, 3, name="t").as_dict("prefix_cache_rate")
    assert row["prefix_cache_rate"] is None
    assert row["prefix_cache_rate_reason"] == INVARIANT_ABOVE_ONE
    assert row["prefix_cache_rate_numerator"] == 9
    assert row["prefix_cache_rate_denominator"] == 3
