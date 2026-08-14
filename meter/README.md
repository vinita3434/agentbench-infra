# meter/

Per-LLM-call instrumentation for the **open-weight serving path**. Entirely
additive: nothing here reads or writes anything the OpenRouter/frontier pipeline
owns, and no existing file was modified to make it work.

Under OpenRouter the serving layer was a black box and the only observable was
whether a task resolved. That question is answered and the task set is fixed. We
own the server now, so the trajectory is the data: how context accumulates turn
over turn, how much of each prompt is served from cache, how the KV pool fills,
how decode throughput and bandwidth move as an episode grows.

## Where it sits

Pi owns the LLM calls. Instrumenting inside it would mean modifying the agent,
which breaks the property the whole benchmark rests on — that the harness is
identical across models. So the meter sits on the wire:

```
Pi  ──►  meter proxy (:8100)  ──►  SGLang (:30000)
                 │
                 └──►  results/serving/<run_label>/<task_id>/turns.jsonl
```

The proxy forwards requests verbatim, relays streamed chunks as they arrive, and
records on the side. Point Pi's `baseUrl` at it and nothing about the agent
changes.

## The five metrics

| metric | scope | source |
|---|---|---|
| KV pool utilization | server, point-in-time | Prometheus gauge, sampled before and after each call |
| Prefix cache rate | **per request** | `cached_tokens / prompt_tokens` from the response's `usage` |
| Weighted prefix efficiency | **per request** | `cached_within_episode / reusable_prefix_tokens` |
| Decode throughput | per request | `(completion_tokens − 1) / (last_token − first_token)` |
| MBU | per request | `bytes_per_token × decode_tps / peak_bandwidth` |

### Why weighted efficiency is the one that matters

A low cache rate has two causes that the rate alone cannot separate:

- **structural non-reuse** — the agent rewrote its context, so little was
  eligible. `reusable` is small, efficiency stays near 1. The server did
  everything it could.
- **eviction loss** — a lot was eligible and the server dropped it anyway.
  `reusable` is large, efficiency well below 1. That is a serving problem: KV
  pressure, radix-tree eviction, batch interference.

Same 10% cache rate, opposite conclusions. Eligibility is computed from the
conversation — the leading run of byte-identical messages between this request
and the previous one — not from the server.

### Cached tokens are attributed, not rejected

`cached_tokens` is everything the server had, from **any** source.
`reusable_prefix_tokens` is only what **this episode's previous turn** made
eligible. They differ legitimately: the system prompt is identical across all
tasks, so a warm server carries prefix across task boundaries.

SGLang's radix cache always matches a contiguous prefix from token 0, so both
are prefix lengths on the same axis and split cleanly:

```
cached_within_episode = min(cached, reusable)       ← server kept what we gave it
cached_beyond_episode = max(0, cached − reusable)   ← carried in from earlier tasks
       (the two sum back to cached_tokens — nothing is discarded)

weighted_prefix_efficiency = cached_within_episode / reusable_prefix_tokens
```

Efficiency uses `within` only: tokens carried in from an earlier task were never
at risk of eviction here, so counting them would flatter the server. `min()`
also bounds the ratio to [0,1] structurally.

Cache rate keeps counting **everything** — it is a work-avoided metric, and
prefill genuinely skipped is genuinely saved regardless of who cached it.

**The cache is never flushed between episodes.** Cross-episode reuse is real
benefit and we want to measure it, so the server is left to behave naturally.

### Decode excludes prefill, deliberately

Request wall time is prefill + decode. In an agentic episode the prompt grows
every turn while replies do not, so total-time throughput drifts downward turn
over turn even at constant generation speed. That drift is prompt growth, not
the serving stack. Separating them needs the first-token timestamp, so **decode
throughput requires streaming**; a non-streamed response records it as absent,
never as an approximation from total time.

## The ratio-above-1 bug

A prefix cache rate above 1 is a bug in the metric, not a property of the run.
It comes from scope mixing: a server-wide cumulative counter (millions of tokens
since boot) used as the numerator over one request's prompt tokens.

Three defences:

1. **Separate types.** `RequestUsage` and `ServerSnapshot` are distinct classes
   with a `scope` field. Per-request computations call `require_request_scope()`,
   which raises `ScopeMixingError` on a server-wide object.
2. **Bounded ratios.** `bounded_ratio()` refuses to return a value above 1. It
   logs an error and returns `Ratio(value=None, reason="invariant:ratio_above_one")`
   with both inputs attached, so the mix-up is diagnosable from the row alone.
3. **Regression tests.** `test_ratios.py::test_ratio_above_one_yields_no_value`
   reproduces the original failure with realistic numbers.

A gap is visibly missing. A 1.4 cache rate looks like a finding.

## Absent is not zero

Every metric is `Optional` and every non-value carries a machine-readable reason:

| reason | meaning |
|---|---|
| `undefined:zero_denominator` | nothing was reusable — turn 1 of an episode |
| `undefined:missing_input` | the server did not report a needed field |
| `undefined:not_streamed` | no first-token boundary, so no decode window |
| `undefined:too_few_output_tokens` | ≤1 output token; no decode rate exists |
| `invariant:ratio_above_one` | scope mixing — see above |
| `invariant:reusable_exceeds_prompt` | eligible > prompt; mixed-up requests |

**Turn 1 is the trap.** Nothing was reusable, so efficiency is *undefined*, not
0%. Recording 0.0 would drag every episode average down with a row that had
nothing to hit. Filter on `weighted_prefix_efficiency_reason IS NULL` before
aggregating.

## Running it

```bash
# 1. SGLang is already up on :30000 (serve/launch.sh)

# 2. Start the meter
METER_RUN_LABEL=qwen3-coder-30b \
METER_TASK_ID=langchain-ai__langchain-7653 \
METER_GPU=H100_SXM \
METER_QUANTIZATION=fp8 \
METER_ACTIVE_PARAMS=3.3e9 \
METER_TOKENIZER=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  python -m meter.proxy

# 3. Write the run-level record (same env)
python -m meter.run_meta

# 4. Point Pi at the proxy instead of :30000 — baseUrl http://localhost:8100/v1
```

Set `METER_ACTIVE_PARAMS` to **active**, not total, parameters. qwen3-coder-30b
is 30.5B total but 3.3B active; only routed experts are read per decoded token,
so using the total is a ~9× error in MBU. The bound check catches the resulting
>1 values, but the fix is the right constant.

Both MBU inputs and the tokenizer are checked at startup and warned about
**before** any GPU time is spent.

## Output

```
results/serving/<run_label>/<task_id>/
    turns.jsonl        one raw row per LLM call
    run_meta.json      model, quantization, SGLang version, launch flags, GPU, harness SHA
    meter_errors.log   metering failures (never fatal)
```

**Raw rows only.** No aggregation in the collection path — that is a separate
offline step over the JSONL. Collecting and summarizing in one pass makes it
impossible to recompute a summary after finding a bug in it, and there is always
a bug in it.

## Aggregating (offline)

```bash
python -m meter.aggregate results/serving/qwen3-coder-30b/<task>/turns.jsonl
python -m meter.aggregate --sweep results/serving/qwen3-coder-30b --csv sweep.csv
```

The headline number is **pooled**, not a mean of per-turn ratios:

```
weighted_prefix_efficiency = Σ cached_within_episode / Σ reusable_prefix_tokens
```

This is algebraically the reusable-weighted mean of the per-turn efficiencies —
so it needs no per-turn division and no weight normalisation to get wrong:

```
Σ (efficiency_n × reusable_n)     Σ (cached_within_n/reusable_n × reusable_n)
─────────────────────────────  =  ───────────────────────────────────────────  =  Σ within / Σ reusable
       Σ reusable_n                            Σ reusable_n
```

Weighting by reusable tokens is the point: a miss on turn 40 with 40k eligible
tokens is a far bigger loss than a miss on turn 2 with 1.2k, and a plain mean
would call them equal. Turn 1 handles itself — `reusable = 0` contributes 0 to
both sums, so a row with nothing to hit cannot drag the average down.

`unweighted_mean_efficiency` is emitted alongside for contrast. A large gap
means the losses are concentrated in the turns carrying the most tokens.

Also reported: `recomputed_tokens` (`Σ reusable − Σ within`) — the absolute
wasted prefill, which converts to GPU-seconds and is often more actionable than
the ratio. Every aggregate carries the row count it came from, so a median over
3 of 60 rows is never mistaken for one over 60. Rows with `reusable: null` are
dropped entirely, never zeroed.

Rows are flushed and `fsync`ed individually: an episode that dies mid-trajectory
still leaves every completed turn on disk.

## Files

| file | role |
|---|---|
| `scopes.py` | `RequestUsage` vs `ServerSnapshot`; response and Prometheus parsing |
| `ratios.py` | bounded ratios, reason vocabulary, scope guard |
| `prefix.py` | cache rate, weighted efficiency, eligibility from the conversation |
| `perf.py` | decode throughput, MBU |
| `record.py` | row builder (never raises) and append-only JSONL writer |
| `proxy.py` | the metered client the harness points at |
| `run_meta.py` | run-level provenance writer |
| `tokens.py` | chat-templated prefix counting; no heuristic fallback |
| `aggregate.py` | offline episode/sweep summaries (never runs during collection) |
| `config.py` | all constants, explicit; env or file |

```bash
python -m pytest meter/tests -q     # 57 tests
```

## Known limits

- **MBU assumes batch 1.** With concurrent requests the weight read is amortized
  across the batch, so per-request MBU understates true utilization. Rows carry
  `server_before_running_requests` so an offline step can account for it.
- **Streaming usage needs `stream_options.include_usage`.** Without it a streamed
  response carries no token counts, and token-dependent metrics are absent.
  Chunk counts are never substituted for token counts.
- **One episode at a time.** The proxy keeps previous-turn context in memory and
  is scoped to a single task. Concurrent episodes need one proxy each, on
  different ports.
