#!/usr/bin/env python3
"""Prefix reuse: how much was cached, and how much of what *could* be was.

Two different questions, and the difference is the point of this module.

  prefix_cache_rate = cached_tokens / prompt_tokens
      Of everything this request sent, what fraction did the server already
      have? Falls whenever the prompt changes early, even if the cache is
      behaving perfectly. On turn 1 of an episode it is ~0 by construction.

  weighted_prefix_efficiency = cached_tokens / reusable_prefix_tokens
      Of the tokens that were *eligible* for reuse -- the part of this prompt
      that is byte-identical to the previous request's context -- what fraction
      did the server actually serve from cache?

Why the second one matters more: a low cache rate has two completely different
causes that the rate alone cannot separate.

  structural non-reuse   the agent rewrote its context, so little was eligible.
                         reusable is small; efficiency stays near 1. Nothing is
                         wrong with the server; the harness changed the prompt.
  eviction loss          a lot was eligible and the server dropped it anyway.
                         reusable is large, efficiency well below 1. This is a
                         serving problem: KV pool pressure, radix tree eviction,
                         batch interference.

Same low rate, opposite conclusions. Efficiency is what tells them apart.

Eligibility is computed from the conversation, not from the server: the client
sees request N-1's messages plus the assistant reply, and request N's messages.
The common leading run of *identical* messages is what a prefix cache could
have matched. Counting those tokens needs a tokenizer for the served model; if
none is configured, reusable_prefix_tokens is None and efficiency is undefined.
It is never estimated -- a guessed denominator produces a confidently wrong
efficiency, which is worse than an absent one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from .ratios import Ratio, bounded_ratio, require_request_scope
from .scopes import RequestUsage

TokenCounter = Callable[[str], int]


def _canonical(message: Any) -> str:
    """Stable serialization for equality testing between turns.

    Sorted keys so an unordered dict does not read as a different message.
    Equality here is deliberately strict: a prefix cache matches on exact token
    prefixes, so anything short of identical content is not eligible.
    """
    try:
        return json.dumps(message, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(message)


def common_message_prefix(
    previous: Optional[Sequence[Any]], current: Optional[Sequence[Any]]
) -> int:
    """Number of leading messages identical in both requests.

    Returns 0 when either side is missing -- on the first turn of an episode
    there is no previous context, so nothing was eligible for reuse. That 0 is a
    real structural fact, and downstream it makes the efficiency denominator 0,
    which ratios.py reports as undefined rather than as a 0% hit rate.
    """
    if not previous or not current:
        return 0
    n = 0
    for prev_msg, cur_msg in zip(previous, current):
        if _canonical(prev_msg) != _canonical(cur_msg):
            break
        n += 1
    return n


def _message_text(message: Any) -> str:
    """Text of a chat message, for token counting.

    Content may be a string or a list of typed parts. Non-text parts contribute
    no countable text here; a multimodal prompt would need a different counter,
    and silently counting nothing for an image is better than inventing a number.
    """
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
    # Tool calls are part of the prompt the server caches, so they count.
    for key in ("tool_calls", "function_call"):
        if message.get(key):
            parts.append(_canonical(message[key]))
    return "\n".join(parts)


def reusable_prefix_tokens(
    previous_messages: Optional[Sequence[Any]],
    current_messages: Optional[Sequence[Any]],
    token_counter: Optional[TokenCounter] = None,
    prefix_counter: Optional[Callable[[List[Any]], Optional[int]]] = None,
) -> Optional[int]:
    """Tokens eligible for prefix reuse on this request.

    `prefix_counter` is preferred: it counts the matched messages as one
    chat-templated sequence, which is what the server tokenizes. `token_counter`
    is the per-message fallback and under-counts by the template's overhead.

    None when neither is available: the count would be a guess, and a guessed
    denominator makes efficiency look precise while being wrong. 0 is returned
    only when genuinely nothing matched -- a real structural zero.
    """
    matched = common_message_prefix(previous_messages, current_messages)
    if matched == 0:
        return 0
    prefix = list(current_messages)[:matched]
    if prefix_counter is not None:
        try:
            return prefix_counter(prefix)
        except Exception:  # a broken counter must not take the episode down
            return None
    if token_counter is None:
        return None
    try:
        return sum(token_counter(_message_text(m)) for m in prefix)
    except Exception:
        return None


@dataclass(frozen=True)
class PrefixMetrics:
    """Both prefix views for one request, plus the inputs behind them."""

    cache_rate: Ratio
    weighted_efficiency: Ratio
    matched_messages: Optional[int] = None
    reusable_tokens: Optional[int] = None
    cached_within_episode: Optional[int] = None
    cached_beyond_episode: Optional[int] = None
    reuse_scope: Optional[str] = None

    def as_dict(self) -> dict:
        out = {}
        out.update(self.cache_rate.as_dict("prefix_cache_rate"))
        out.update(self.weighted_efficiency.as_dict("weighted_prefix_efficiency"))
        out["prefix_matched_messages"] = self.matched_messages
        out["prefix_reusable_tokens"] = self.reusable_tokens
        # The two halves sum back to cached_tokens. Kept separate because they
        # answer different questions: retention vs cross-episode carryover.
        out["cached_within_episode"] = self.cached_within_episode
        out["cached_beyond_episode"] = self.cached_beyond_episode
        out["prefix_reuse_scope"] = self.reuse_scope
        return out


def compute_prefix_metrics(
    usage: RequestUsage,
    previous_messages: Optional[Sequence[Any]] = None,
    current_messages: Optional[Sequence[Any]] = None,
    token_counter: Optional[TokenCounter] = None,
    prefix_counter: Optional[Callable[[List[Any]], Optional[int]]] = None,
) -> PrefixMetrics:
    """Both prefix metrics for a single request.

    `usage` must be per-request. That is asserted rather than assumed: passing a
    server-wide cumulative hit counter here is the documented way this metric
    goes above 1, and it is the one failure this module is built to refuse.
    """
    require_request_scope(usage, name="compute_prefix_metrics")

    # Work-avoided view: every cached token counts, whatever put it there.
    cache_rate = bounded_ratio(
        usage.cached_tokens, usage.prompt_tokens, name="prefix_cache_rate"
    )

    matched = common_message_prefix(previous_messages, current_messages)
    reusable = reusable_prefix_tokens(
        previous_messages, current_messages, token_counter, prefix_counter
    )

    cached = usage.cached_tokens
    within = beyond = None
    scope = None

    if cached is not None and reusable is not None:
        # SGLang matches a contiguous prefix from token 0, so `cached` and
        # `reusable` are lengths on the same axis and split cleanly. The halves
        # sum back to `cached`: nothing is discarded, only attributed.
        within = min(cached, reusable)
        beyond = max(0, cached - reusable)
        if cached == 0:
            scope = "none"
        elif beyond > 0:
            # The server matched more than this episode supplied -- the excess
            # came from an earlier episode (typically the shared system prompt).
            # Real caching benefit, and invisible before this split.
            scope = "spans_episodes"
        else:
            scope = "within_only"

    # Eligible tokens can never exceed the prompt itself. If they do, the two
    # numbers describe different requests -- refuse rather than emit.
    if (
        reusable is not None
        and usage.prompt_tokens is not None
        and reusable > usage.prompt_tokens
    ):
        efficiency = Ratio(
            None, "invariant:reusable_exceeds_prompt", cached, reusable
        )
    else:
        # Retention view: only what THIS episode made eligible. Tokens carried
        # in from an earlier task were never at risk of eviction here, so
        # counting them would flatter the server. min() also bounds this to
        # [0,1] structurally, so it can never exceed 1 for this reason.
        efficiency = bounded_ratio(
            within, reusable, name="weighted_prefix_efficiency"
        )

    return PrefixMetrics(
        cache_rate=cache_rate,
        weighted_efficiency=efficiency,
        matched_messages=matched,
        reusable_tokens=reusable,
        cached_within_episode=within,
        cached_beyond_episode=beyond,
        reuse_scope=scope,
    )
