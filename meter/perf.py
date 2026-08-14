#!/usr/bin/env python3
"""Decode throughput and model bandwidth utilization, per request.

Decode throughput excludes prefill on purpose. A request's wall time is
prefill + decode, and prefill scales with prompt length while decode scales
with output length. In an agentic episode the prompt grows every turn and the
replies do not, so total-time throughput drifts downward turn over turn even
when generation speed is unchanged. That drift is prompt growth, not the
serving stack, and it hides what we are actually trying to see.

Separating them needs the first-token timestamp, which needs streaming. A
non-streamed response yields decode_tokens_per_s = None with a reason -- not an
approximation from total time, which would silently be the wrong metric.

MBU (model bandwidth utilization) is the share of the GPU's memory bandwidth
that decode is actually using:

    achieved_bytes_per_s = bytes_moved_per_token * decode_tokens_per_s
    MBU                  = achieved_bytes_per_s / peak_bandwidth_bytes_per_s

At batch 1, decode is memory-bound: every token reads the full weight set, so
bytes_moved_per_token is the model's weight footprint. That is a property of
the deployment (parameter count x bytes per parameter after quantization), and
peak bandwidth is a property of the GPU. Both must be configured explicitly --
neither is inferred, because a wrong constant here produces a plausible-looking
MBU that is simply false. Missing config gives None, not a guess.

Batch > 1 caveat, recorded rather than corrected: with concurrent requests the
weight read is amortized across the batch, so per-request MBU computed this way
understates true utilization. The row carries the server's concurrency at
sample time so an offline step can account for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ratios import Ratio, bounded_ratio

# Reasons specific to timing, kept stable for downstream filtering.
NO_STREAM = "undefined:not_streamed"
NO_DECODE_WINDOW = "undefined:no_decode_window"
TOO_FEW_TOKENS = "undefined:too_few_output_tokens"


@dataclass(frozen=True)
class Timing:
    """Timestamps captured around one completion (seconds, monotonic)."""

    started_at: Optional[float] = None
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def ttft_s(self) -> Optional[float]:
        """Prefill + queue + first decode step, as the client observes it."""
        if self.started_at is None or self.first_token_at is None:
            return None
        return max(0.0, self.first_token_at - self.started_at)

    @property
    def decode_s(self) -> Optional[float]:
        """First token to last token: the decode window, prefill excluded."""
        if self.first_token_at is None or self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.first_token_at)

    @property
    def total_s(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.started_at)


def decode_tokens_per_s(
    completion_tokens: Optional[int], timing: Timing
) -> tuple[Optional[float], Optional[str]]:
    """Output tokens per second during decode only.

    The first token is excluded from the numerator: it is produced at the end of
    prefill, and the decode window starts when it arrives. Counting it would
    credit decode with a token it did not generate in that window. So a
    single-token completion has no measurable decode rate at all.
    """
    if completion_tokens is None:
        return None, "undefined:missing_input"
    if timing.first_token_at is None:
        return None, NO_STREAM
    decode_s = timing.decode_s
    if decode_s is None:
        return None, NO_DECODE_WINDOW
    if completion_tokens <= 1:
        return None, TOO_FEW_TOKENS
    if decode_s <= 0:
        # Every token in one clock tick: real, but not a rate we can divide for.
        return None, NO_DECODE_WINDOW
    return (completion_tokens - 1) / decode_s, None


def model_bytes_per_token(
    param_count: Optional[float], bytes_per_param: Optional[float]
) -> Optional[float]:
    """Weight bytes read per decoded token at batch 1.

    For an MoE model this should be the *active* parameter count, not the total:
    only the routed experts are read per token. qwen3-coder-30b is 30.5B total
    but 3.3B active, and using the wrong one is a ~9x error in MBU.
    """
    if param_count is None or bytes_per_param is None:
        return None
    if param_count <= 0 or bytes_per_param <= 0:
        return None
    return param_count * bytes_per_param


def mbu(
    decode_tps: Optional[float],
    bytes_per_token: Optional[float],
    peak_bandwidth_bytes_per_s: Optional[float],
) -> Ratio:
    """Achieved memory bandwidth as a fraction of the GPU's peak.

    Bounded like any other rate: above 1 is impossible and means a constant is
    wrong (usually total instead of active parameters, or the wrong GPU's peak),
    so it is reported as an invariant violation rather than plotted.
    """
    if decode_tps is None or bytes_per_token is None:
        return Ratio(None, "undefined:missing_input", None, peak_bandwidth_bytes_per_s)
    achieved = decode_tps * bytes_per_token
    return bounded_ratio(achieved, peak_bandwidth_bytes_per_s, name="mbu")


@dataclass(frozen=True)
class PerfMetrics:
    decode_tokens_per_s: Optional[float] = None
    decode_tokens_per_s_reason: Optional[str] = None
    ttft_s: Optional[float] = None
    decode_s: Optional[float] = None
    total_s: Optional[float] = None
    mbu: Ratio = Ratio()
    bytes_per_token: Optional[float] = None

    def as_dict(self) -> dict:
        out = {
            "decode_tokens_per_s": self.decode_tokens_per_s,
            "decode_tokens_per_s_reason": self.decode_tokens_per_s_reason,
            "ttft_s": self.ttft_s,
            "decode_s": self.decode_s,
            "total_s": self.total_s,
            "mbu_bytes_per_token": self.bytes_per_token,
        }
        out.update(self.mbu.as_dict("mbu"))
        return out


def compute_perf(
    completion_tokens: Optional[int],
    timing: Timing,
    param_count: Optional[float] = None,
    bytes_per_param: Optional[float] = None,
    peak_bandwidth_bytes_per_s: Optional[float] = None,
) -> PerfMetrics:
    tps, reason = decode_tokens_per_s(completion_tokens, timing)
    bpt = model_bytes_per_token(param_count, bytes_per_param)
    return PerfMetrics(
        decode_tokens_per_s=tps,
        decode_tokens_per_s_reason=reason,
        ttft_s=timing.ttft_s,
        decode_s=timing.decode_s,
        total_s=timing.total_s,
        mbu=mbu(tps, bpt, peak_bandwidth_bytes_per_s),
        bytes_per_token=bpt,
    )
