#!/usr/bin/env python3
"""Bounded ratios, and an explicit vocabulary for "no number here".

Three outcomes are possible when computing a rate, and collapsing any two of
them loses information that matters downstream:

  a real value        the ratio was computable and is in range
  undefined           the question does not apply to this row -- most often
                      because the denominator is 0. Nothing was reusable, so
                      there is no fraction to report. This is NOT 0.0: averaging
                      a first turn in as a 0% cache hit would drag every mean
                      down with a row that had nothing to hit.
  invariant violated  the ratio came out > 1, which is never a property of a
                      run. It means the numerator and denominator came from
                      different scopes. It must be loud and must not be plotted.

`Ratio` carries the outcome and, when there is no value, a machine-readable
reason. Callers write `ratio.value` (None when absent) and `ratio.reason` into
the row, so an offline aggregation step can filter on reason instead of trying
to distinguish 0.0-the-measurement from 0.0-the-default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("meter.ratios")

# Reasons a ratio has no value. Stable strings -- they end up in the data.
UNDEFINED_ZERO_DENOMINATOR = "undefined:zero_denominator"
UNDEFINED_MISSING_INPUT = "undefined:missing_input"
INVARIANT_ABOVE_ONE = "invariant:ratio_above_one"
INVARIANT_NEGATIVE = "invariant:negative_input"


class ScopeMixingError(ValueError):
    """Raised when a per-request ratio is handed a non-per-request numerator.

    Kept as a hard exception (rather than a quiet None) for programming errors
    that a test can catch. Runtime collection wraps calls so an episode never
    dies from it -- see record.py.
    """


@dataclass(frozen=True)
class Ratio:
    """A ratio that may legitimately have no value."""

    value: Optional[float] = None
    reason: Optional[str] = None
    numerator: Optional[float] = None
    denominator: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.value is not None

    def as_dict(self, prefix: str) -> dict:
        """Flatten into row fields: <prefix>, <prefix>_reason, and the inputs.

        The inputs travel with the ratio on purpose: when a value is missing or
        wrong, the two numbers that produced it are the first thing anyone asks
        for, and reconstructing them later from separate columns is guesswork.
        """
        return {
            prefix: self.value,
            f"{prefix}_reason": self.reason,
            f"{prefix}_numerator": self.numerator,
            f"{prefix}_denominator": self.denominator,
        }


def bounded_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
    *,
    name: str,
    strict: bool = True,
) -> Ratio:
    """numerator/denominator, constrained to [0, 1], with explicit non-values.

    `strict` controls what happens above 1. Default True: log an error and
    return a valued-less Ratio flagged INVARIANT_ABOVE_ONE. The number is never
    returned, because a plot containing it is worse than a gap -- a gap is
    visibly missing, a 1.4 cache rate looks like a finding.
    """
    if numerator is None or denominator is None:
        return Ratio(None, UNDEFINED_MISSING_INPUT, numerator, denominator)

    if numerator < 0 or denominator < 0:
        log.error("%s: negative input (num=%s den=%s)", name, numerator, denominator)
        return Ratio(None, INVARIANT_NEGATIVE, numerator, denominator)

    if denominator == 0:
        # Not a zero rate -- an inapplicable question. First turn of an episode
        # has no prior context, so nothing *could* have been reused.
        return Ratio(None, UNDEFINED_ZERO_DENOMINATOR, numerator, denominator)

    value = numerator / denominator

    if value > 1.0:
        log.error(
            "%s: ratio %.4f > 1 (num=%s den=%s). This is a metric bug, not a run "
            "property -- almost always a server-wide cumulative counter used as a "
            "per-request numerator. Emitting no value.",
            name, value, numerator, denominator,
        )
        if strict:
            return Ratio(None, INVARIANT_ABOVE_ONE, numerator, denominator)

    return Ratio(value, None, numerator, denominator)


def require_request_scope(obj, *, name: str) -> None:
    """Fail loudly if `obj` is not per-request.

    The type system already prevents passing a ServerSnapshot where a
    RequestUsage belongs, but values get unpacked and passed around as plain
    ints. This is the runtime backstop for the path where someone reaches into
    a snapshot and hands the number over by hand.
    """
    scope = getattr(obj, "scope", None)
    if scope != "request":
        raise ScopeMixingError(
            f"{name} requires a per-request source, got scope={scope!r} "
            f"({type(obj).__name__}). Server-wide counters are cumulative since "
            f"process start and are never a per-request numerator."
        )
