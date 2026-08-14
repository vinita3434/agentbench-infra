#!/usr/bin/env python3
"""Exact token counting for the reusable-prefix denominator.

Weighted prefix efficiency divides cached tokens by *eligible* tokens, and the
denominator has to be counted with the same tokenizer the server uses. An
approximation (chars/4, whitespace splitting) would produce an efficiency that
looks precise and is wrong by tens of percent -- and unlike an absent metric, a
wrong one is invisible in a plot.

So there is exactly one supported path -- the served model's own tokenizer, via
transformers -- and no fallback. If it cannot be loaded, this returns None, the
denominator is None, and efficiency is reported absent with a reason. That is
the intended behaviour, not a degradation: prefix cache rate and every other
metric still record normally.

    METER_TOKENIZER=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 python -m meter.proxy

Cache the tokenizer locally on the pod (HF_HOME) so it is not re-fetched per
episode.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger("meter.tokens")

TokenCounter = Callable[[str], int]


def load_token_counter(identifier: Optional[str]) -> Optional[TokenCounter]:
    """A callable counting tokens with the served model's tokenizer, or None.

    None is a legitimate outcome and is handled everywhere downstream. It is
    never replaced with a heuristic counter -- see the module docstring.
    """
    if not identifier:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.warning(
            "transformers not installed; reusable-prefix tokens will be absent "
            "and weighted_prefix_efficiency undefined. pip install transformers"
        )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(identifier, trust_remote_code=False)
    except Exception as exc:
        log.warning("could not load tokenizer %r: %s", identifier, exc)
        return None

    def count(text: str) -> int:
        # add_special_tokens=False: BOS/EOS belong to the full sequence, not to
        # each message, and adding them per message would inflate the
        # denominator and depress efficiency.
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    log.info("token counter ready: %s", identifier)
    return count


PrefixCounter = Callable[[list], Optional[int]]


def load_prefix_counter(identifier: Optional[str]) -> Optional[PrefixCounter]:
    """Count tokens of a *message prefix* the way the server actually sees it.

    The server never tokenizes raw message text. It applies the model's chat
    template first, which inserts role markers and special tokens around every
    message. Counting the bare text therefore under-counts, and an under-counted
    denominator makes real reuse look like cross-episode carryover -- the exact
    confusion the decomposition is meant to resolve.

    add_generation_prompt=False because a prefix sits mid-conversation; the
    trailing "assistant:" cue belongs only at the end of a full prompt.
    """
    if not identifier:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.warning(
            "transformers not installed; reusable-prefix tokens will be absent "
            "and weighted_prefix_efficiency undefined. pip install transformers"
        )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(identifier, trust_remote_code=False)
    except Exception as exc:
        log.warning("could not load tokenizer %r: %s", identifier, exc)
        return None

    has_template = getattr(tokenizer, "chat_template", None) is not None

    def count_prefix(messages: list) -> Optional[int]:
        if not messages:
            return 0
        try:
            if has_template:
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
                return len(ids)
            # No template published for this model: fall back to summing the
            # message texts, and say so -- this under-counts by the template's
            # overhead, so efficiency will read slightly low.
            text = "\n".join(
                m.get("content") if isinstance(m.get("content"), str) else ""
                for m in messages if isinstance(m, dict)
            )
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:
            log.debug("prefix token count failed: %s", exc)
            return None

    log.info(
        "prefix counter ready: %s (chat_template=%s)",
        identifier, "yes" if has_template else "NO -- counts will under-read",
    )
    return count_prefix
