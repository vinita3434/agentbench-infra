#!/usr/bin/env python3
"""Offline aggregation: pooled weighting, and absent-not-zero at episode scale."""
import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from meter.aggregate import read_rows, summarize, summarize_file  # noqa: E402


def row(**kw):
    """A turn row with the fields the aggregator reads, all absent by default."""
    base = {
        "run_label": "qwen3-coder-30b", "task_id": "t1",
        "prompt_tokens": None, "cached_tokens": None,
        "prefix_reusable_tokens": None, "cached_within_episode": None,
        "cached_beyond_episode": None, "prefix_reuse_scope": None,
        "weighted_prefix_efficiency": None, "decode_tokens_per_s": None,
        "mbu": None, "ttft_s": None, "server_after_kv_pool_utilization": None,
    }
    base.update(kw)
    return base


# The worked example: efficiency 1.0, 1.0, 0.5, 1.0 with growing prefixes.
EPISODE = [
    row(call_index=1, prompt_tokens=1000, cached_tokens=0,
        prefix_reusable_tokens=0, cached_within_episode=0, cached_beyond_episode=0,
        weighted_prefix_efficiency=None),
    row(call_index=2, prompt_tokens=2000, cached_tokens=1200,
        prefix_reusable_tokens=1200, cached_within_episode=1200, cached_beyond_episode=0,
        weighted_prefix_efficiency=1.0),
    row(call_index=3, prompt_tokens=3000, cached_tokens=2200,
        prefix_reusable_tokens=2200, cached_within_episode=2200, cached_beyond_episode=0,
        weighted_prefix_efficiency=1.0),
    row(call_index=4, prompt_tokens=4000, cached_tokens=1600,
        prefix_reusable_tokens=3200, cached_within_episode=1600, cached_beyond_episode=0,
        weighted_prefix_efficiency=0.5),
    row(call_index=5, prompt_tokens=5000, cached_tokens=4000,
        prefix_reusable_tokens=4000, cached_within_episode=4000, cached_beyond_episode=0,
        weighted_prefix_efficiency=1.0),
]


def test_pooled_equals_reusable_weighted_mean():
    """Σ within / Σ reusable == Σ(eff_n × reusable_n) / Σ reusable_n."""
    s = summarize(EPISODE)
    assert s["sum_cached_within_episode"] == 9000
    assert s["sum_reusable_tokens"] == 10600
    assert s["weighted_prefix_efficiency"] == pytest.approx(9000 / 10600)

    manual = sum(r["weighted_prefix_efficiency"] * r["prefix_reusable_tokens"]
                 for r in EPISODE if r["weighted_prefix_efficiency"] is not None)
    assert s["weighted_prefix_efficiency"] == pytest.approx(manual / 10600)


def test_weighting_actually_changes_the_answer():
    """If it matched the plain mean, the weighting would be pointless."""
    s = summarize(EPISODE)
    assert s["weighted_prefix_efficiency"] == pytest.approx(0.849, abs=1e-3)
    assert s["unweighted_mean_efficiency"] == pytest.approx(0.875)
    # Lower, because the miss landed on the turn carrying the most tokens.
    assert s["weighted_prefix_efficiency"] < s["unweighted_mean_efficiency"]


def test_turn_one_self_excludes():
    """reusable=0 contributes 0 to both sums, so it cannot drag the mean down."""
    without_turn1 = summarize(EPISODE[1:])
    assert (without_turn1["weighted_prefix_efficiency"]
            == pytest.approx(summarize(EPISODE)["weighted_prefix_efficiency"]))


def test_recomputed_tokens_is_the_absolute_loss():
    s = summarize(EPISODE)
    assert s["recomputed_tokens"] == 10600 - 9000   # 1600, all from turn 4


def test_rows_without_a_tokenizer_are_excluded_not_zeroed():
    """reusable=None must not enter the denominator as 0."""
    rows = EPISODE + [
        row(call_index=6, prompt_tokens=6000, cached_tokens=5000,
            prefix_reusable_tokens=None, cached_within_episode=None),
    ]
    s = summarize(rows)
    assert s["turns"] == 6
    assert s["turns_with_measured_reusable"] == 5          # the None row dropped
    assert s["sum_reusable_tokens"] == 10600               # unchanged
    assert s["weighted_prefix_efficiency"] == pytest.approx(9000 / 10600)


def test_cache_rate_pooled_counts_all_cached_tokens():
    """Work-avoided view: cross-episode hits count here, unlike efficiency."""
    rows = [
        row(prompt_tokens=1000, cached_tokens=800,
            prefix_reusable_tokens=500, cached_within_episode=500,
            cached_beyond_episode=300, prefix_reuse_scope="spans_episodes"),
    ]
    s = summarize(rows)
    assert s["prefix_cache_rate_pooled"] == pytest.approx(0.8)   # all 800
    assert s["weighted_prefix_efficiency"] == pytest.approx(1.0)  # only the 500
    assert s["cached_beyond_episode_total"] == 300
    assert s["turns_spanning_episodes"] == 1


def test_empty_denominator_is_undefined_not_zero():
    rows = [row(prompt_tokens=500, cached_tokens=0,
                prefix_reusable_tokens=0, cached_within_episode=0)]
    s = summarize(rows)
    assert s["weighted_prefix_efficiency"] is None
    assert s["weighted_prefix_efficiency_reason"] == "undefined:zero_denominator"
    assert s["recomputed_tokens"] is None


def test_counts_travel_with_every_aggregate():
    """A median over 2 rows must not look like a median over 60."""
    rows = [
        row(decode_tokens_per_s=100.0, mbu=0.1),
        row(decode_tokens_per_s=200.0, mbu=None),
        row(decode_tokens_per_s=None, mbu=None),
    ]
    s = summarize(rows)
    assert s["decode_tps_median"] == pytest.approx(150.0)
    assert s["turns_with_decode_rate"] == 2
    assert s["turns_with_mbu"] == 1
    assert s["turns"] == 3


def test_missing_reasons_are_counted():
    rows = [
        row(weighted_prefix_efficiency_reason="undefined:zero_denominator"),
        row(decode_tokens_per_s_reason="undefined:not_streamed"),
        row(decode_tokens_per_s_reason="undefined:not_streamed"),
    ]
    s = summarize(rows)
    assert s["excluded_rows"]["decode_tokens_per_s_reason=undefined:not_streamed"] == 2


def test_empty_episode_does_not_crash():
    assert summarize([])["turns"] == 0


def test_truncated_jsonl_is_tolerated():
    """An episode killed mid-write still yields its completed turns."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "turns.jsonl"
        path.write_text(
            json.dumps(EPISODE[0]) + "\n"
            + json.dumps(EPISODE[1]) + "\n"
            + '{"call_index": 3, "prompt_to'      # torn final line
        )
        assert len(read_rows(path)) == 2


def test_summarize_file_writes_summary_next_to_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "turns.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in EPISODE))
        summarize_file(path)
        written = json.loads((path.parent / "episode_summary.json").read_text())
        assert written["weighted_prefix_efficiency"] == pytest.approx(9000 / 10600)
