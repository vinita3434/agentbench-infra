#!/usr/bin/env python3
"""Prefix logic: eligibility, the two rates, and what they separate."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from meter.prefix import (  # noqa: E402
    common_message_prefix,
    compute_prefix_metrics,
    reusable_prefix_tokens,
)
from meter.ratios import (  # noqa: E402
    INVARIANT_ABOVE_ONE,
    UNDEFINED_ZERO_DENOMINATOR,
    ScopeMixingError,
)
from meter.scopes import RequestUsage, ServerSnapshot, usage_from_response  # noqa: E402

# 4 chars/token: crude, but a *test* counter, never a production default.
WORDS = lambda text: max(1, len(text) // 4)  # noqa: E731


def msg(role, content):
    return {"role": role, "content": content}


TURN1 = [msg("system", "You are a coding agent."), msg("user", "Fix the cache bug.")]
TURN2 = TURN1 + [msg("assistant", "Looking at cache.py"), msg("user", "continue")]


# --- eligibility ----------------------------------------------------------


def test_common_prefix_counts_identical_leading_messages():
    assert common_message_prefix(TURN1, TURN2) == 2


def test_no_previous_context_means_nothing_eligible():
    """Turn 1: nothing could have been reused. Structural, not a cache miss."""
    assert common_message_prefix(None, TURN1) == 0


def test_divergence_stops_the_prefix():
    rewritten = [TURN1[0], msg("user", "Completely different task.")]
    assert common_message_prefix(TURN1, rewritten) == 1


def test_reusable_tokens_none_without_counter():
    """No tokenizer means no denominator -- never an estimate."""
    assert reusable_prefix_tokens(TURN1, TURN2, token_counter=None) is None


def test_reusable_tokens_counted_with_counter():
    assert reusable_prefix_tokens(TURN1, TURN2, token_counter=WORDS) > 0


def test_reusable_is_zero_when_nothing_matches():
    """A real structural zero, distinct from 'could not count'."""
    assert reusable_prefix_tokens(None, TURN1, token_counter=WORDS) == 0


# --- the two rates and what they separate ---------------------------------


def test_cache_rate_and_efficiency_are_different_questions():
    """Little eligible, all of it hit: low rate, perfect efficiency.

    This is structural non-reuse -- the prompt changed, so the cache had little
    to offer. The server did everything it could. Rate alone would look bad.
    """
    usage = RequestUsage(prompt_tokens=1000, completion_tokens=50, cached_tokens=50)
    m = compute_prefix_metrics(usage, TURN1, TURN2, token_counter=lambda t: 25)

    assert m.cache_rate.value == pytest.approx(0.05)     # 50/1000 -- looks poor
    assert m.reusable_tokens == 50                        # 2 matched x 25
    assert m.weighted_efficiency.value == pytest.approx(1.0)   # kept everything
    assert m.reuse_scope == "within_only"


def test_cross_episode_reuse_is_attributed_not_rejected():
    """The server had more than this episode gave it.

    Cached 100 against 50 eligible: the extra 50 came from an earlier episode --
    the shared system prompt surviving across tasks. That is real caching
    benefit, so it is split out rather than discarded as a >1 ratio.
    """
    usage = RequestUsage(prompt_tokens=1000, completion_tokens=50, cached_tokens=100)
    m = compute_prefix_metrics(usage, TURN1, TURN2, token_counter=lambda t: 25)

    assert m.reusable_tokens == 50
    assert m.cached_within_episode == 50
    assert m.cached_beyond_episode == 50
    assert m.reuse_scope == "spans_episodes"
    # Bounded by construction now -- min() makes >1 unreachable here.
    assert m.weighted_efficiency.value == pytest.approx(1.0)
    # The halves sum back to what the server reported. Nothing is dropped.
    assert m.cached_within_episode + m.cached_beyond_episode == usage.cached_tokens


def test_turn_one_on_a_warm_server():
    """Nothing eligible, yet tokens cached: all of it is cross-episode.

    Efficiency stays undefined (no denominator), but the carryover is recorded
    instead of vanishing -- this is the shared-system-prompt case made visible.
    """
    usage = RequestUsage(prompt_tokens=500, cached_tokens=180)
    m = compute_prefix_metrics(usage, None, TURN1, token_counter=WORDS)

    assert m.reusable_tokens == 0
    assert m.weighted_efficiency.value is None
    assert m.weighted_efficiency.reason == UNDEFINED_ZERO_DENOMINATOR
    assert m.cached_within_episode == 0
    assert m.cached_beyond_episode == 180
    assert m.reuse_scope == "spans_episodes"


def test_eviction_loss_shows_as_low_efficiency():
    """Lots eligible, little hit: the eviction signal.

    Same 10% cache rate as a structural case would give, but efficiency is 0.125
    -- the server dropped what it could have kept. Opposite conclusion, and only
    efficiency distinguishes them.
    """
    usage = RequestUsage(prompt_tokens=1000, completion_tokens=50, cached_tokens=100)
    m = compute_prefix_metrics(usage, TURN1, TURN2, token_counter=lambda t: 400)

    assert m.cache_rate.value == pytest.approx(0.1)
    assert m.reusable_tokens == 800
    assert m.weighted_efficiency.value == pytest.approx(0.125)


def test_perfect_reuse():
    usage = RequestUsage(prompt_tokens=1000, cached_tokens=800)
    m = compute_prefix_metrics(usage, TURN1, TURN2, token_counter=lambda t: 400)
    assert m.weighted_efficiency.value == pytest.approx(1.0)


def test_first_turn_efficiency_is_undefined_not_zero():
    """The averaging trap: turn 1 must not enter a mean as a 0.

    Nothing was reusable, so there is no fraction. Recording 0.0 would drag
    every episode average down with a row that had nothing to hit.
    """
    usage = RequestUsage(prompt_tokens=500, cached_tokens=0)
    m = compute_prefix_metrics(usage, None, TURN1, token_counter=WORDS)

    assert m.reusable_tokens == 0
    assert m.weighted_efficiency.value is None
    assert m.weighted_efficiency.reason == UNDEFINED_ZERO_DENOMINATOR
    assert m.weighted_efficiency.value != 0.0
    # The cache rate is a genuine 0 on the same row: nothing was cached out of
    # 500 real prompt tokens. Absent and zero, side by side, both correct.
    assert m.cache_rate.value == 0.0


def test_reusable_exceeding_prompt_is_refused():
    """Eligible tokens cannot exceed the prompt -- that is mixed-up requests."""
    usage = RequestUsage(prompt_tokens=100, cached_tokens=50)
    m = compute_prefix_metrics(usage, TURN1, TURN2, token_counter=lambda t: 5000)
    assert m.weighted_efficiency.value is None
    assert m.weighted_efficiency.reason == "invariant:reusable_exceeds_prompt"


def test_server_snapshot_cannot_be_used_as_request_usage():
    """REGRESSION: the scope error that produced rates above 1."""
    with pytest.raises(ScopeMixingError):
        compute_prefix_metrics(ServerSnapshot(cache_hit_rate_serverwide=0.9), TURN1, TURN2)


def test_missing_cached_tokens_is_absent_not_zero():
    """A server that reports no cache field is not a server reporting 0 hits."""
    usage = usage_from_response({"usage": {"prompt_tokens": 900, "completion_tokens": 10}})
    assert usage.cached_tokens is None
    m = compute_prefix_metrics(usage, TURN1, TURN2, token_counter=WORDS)
    assert m.cache_rate.value is None
    assert m.cache_rate.reason == "undefined:missing_input"


def test_row_dict_shape():
    usage = RequestUsage(prompt_tokens=1000, cached_tokens=100)
    row = compute_prefix_metrics(usage, TURN1, TURN2, token_counter=lambda t: 400).as_dict()
    for key in (
        "prefix_cache_rate",
        "prefix_cache_rate_reason",
        "weighted_prefix_efficiency",
        "weighted_prefix_efficiency_reason",
        "prefix_matched_messages",
        "prefix_reusable_tokens",
    ):
        assert key in row
