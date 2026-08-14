#!/usr/bin/env python3
"""A metering proxy: OpenAI-compatible in, OpenAI-compatible out.

Pi owns the LLM calls, and instrumenting inside it would mean modifying the
agent -- which would break the one property the benchmark depends on, that the
harness is identical across models. So the meter sits on the wire instead:

    Pi  ->  this proxy (:8100)  ->  SGLang (:30000)

Point Pi's baseUrl at the proxy and nothing about the agent changes. The proxy
forwards the request verbatim, streams the response back untouched, and records
one row per call on the side.

Transparency is the hard requirement. The body returned to Pi is byte-identical
to what SGLang sent, streaming chunks arrive as they arrive, and any failure in
the metering path is swallowed after logging. If the meter cannot do its job the
episode still runs -- a lost metric is cheap, a lost trajectory is not.

Usage:
    METER_RUN_LABEL=qwen3-coder-30b METER_TASK_ID=<task> \
    METER_GPU=H100_SXM METER_QUANTIZATION=fp8 METER_ACTIVE_PARAMS=3.3e9 \
        python -m meter.proxy

Then point Pi at http://localhost:8100/v1 (see meter/README.md).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from .config import MeterConfig
from .perf import Timing
from .record import RowWriter, build_row
from .scopes import RequestUsage, snapshot_from_prometheus, usage_from_response
from .tokens import load_prefix_counter, load_token_counter

log = logging.getLogger("meter.proxy")

CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")


class EpisodeState:
    """Per-episode memory: the call counter and the previous request's context.

    The previous turn's messages are what makes weighted prefix efficiency
    computable -- eligibility is defined against them. Kept in memory only; the
    rows on disk are the durable artifact.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.call_index = 0
        self.previous_messages: Optional[List[Any]] = None

    def next_index(self) -> int:
        with self.lock:
            self.call_index += 1
            return self.call_index

    def swap_messages(self, new_messages, assistant_message=None):
        """Return the previous context and install the new one.

        The stored context is the request's messages plus the assistant reply,
        because that whole sequence is what the next request will share a prefix
        with -- the server caches what it generated as well as what it was sent.
        """
        with self.lock:
            previous = self.previous_messages
            stored = list(new_messages or [])
            if assistant_message:
                stored.append(assistant_message)
            self.previous_messages = stored
            return previous


def scrape(url: Optional[str], timeout: float):
    """Prometheus scrape that fails soft: any problem yields None."""
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return snapshot_from_prometheus(
                resp.read().decode("utf-8", "replace"), scraped_at=time.time()
            )
    except Exception as exc:
        log.debug("prometheus scrape failed: %s", exc)
        return None


def _sse_payloads(raw: bytes):
    """Yield parsed JSON objects from an SSE stream body."""
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _usage_and_text_from_stream(raw: bytes):
    """Recover usage and assistant text from a streamed response.

    Streaming responses carry usage only if the caller asked for it
    (stream_options.include_usage). When absent, token counts are None rather
    than counted from chunks: a chunk count is not a token count, and a wrong
    prompt_tokens would silently corrupt every ratio on the row.
    """
    usage_obj: Optional[Dict[str, Any]] = None
    text_parts: List[str] = []
    tool_calls: List[Any] = []
    for payload in _sse_payloads(raw):
        if isinstance(payload.get("usage"), dict):
            usage_obj = payload["usage"]
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            if delta.get("tool_calls"):
                tool_calls.append(delta["tool_calls"])
    usage = usage_from_response({"usage": usage_obj}) if usage_obj else RequestUsage()
    assistant = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return usage, assistant


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # injected by serve()
    cfg: MeterConfig
    writer: RowWriter
    state: EpisodeState
    token_counter = None   # per-message fallback
    prefix_counter = None  # chat-templated; preferred. None -> efficiency absent

    def log_message(self, *args):  # keep the proxy quiet; we have our own logs
        pass

    # --- plumbing ---------------------------------------------------------

    def _upstream(self, path: str) -> str:
        base = self.cfg.upstream_base_url.rstrip("/")
        if path.startswith("/v1/"):
            path = path[3:]
        return base + path

    def do_GET(self):
        self._forward(b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(length) if length else b"")

    def _forward(self, body: bytes):
        is_chat = any(self.path.startswith(p) for p in CHAT_PATHS)
        started = time.time()
        first_byte_at: Optional[float] = None

        request_messages = None
        if is_chat and body:
            try:
                request_messages = (json.loads(body) or {}).get("messages")
            except Exception:
                request_messages = None

        server_before = None
        if is_chat and self.cfg.scrape_prometheus:
            server_before = scrape(self.cfg.prometheus_url, self.cfg.prometheus_timeout_s)

        req = urllib.request.Request(
            self._upstream(self.path),
            data=body or None,
            method=self.command,
            headers={
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection")
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                status, headers = resp.status, resp.headers
                chunks: List[bytes] = []
                self.send_response(status)
                for k, v in headers.items():
                    if k.lower() in ("transfer-encoding", "content-length", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()

                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    if first_byte_at is None:
                        first_byte_at = time.time()
                    chunks.append(chunk)
                    # Relay immediately: buffering would distort the very
                    # inter-token timing we are trying to measure.
                    self.wfile.write(chunk)
                    self.wfile.flush()
                raw = b"".join(chunks)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
        except Exception as exc:
            log.error("upstream request failed: %s", exc)
            msg = json.dumps({"error": {"message": f"meter proxy upstream error: {exc}"}}).encode()
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(msg)
            except Exception:
                pass
            return

        if not is_chat:
            return

        finished = time.time()
        timing = Timing(started_at=started, first_token_at=first_byte_at, finished_at=finished)
        # Metering is strictly after the client has its bytes, and cannot raise
        # into the response path.
        try:
            self._record(raw, timing, request_messages, server_before)
        except Exception as exc:
            self.writer.log_error(f"record failed: {exc}")

    # --- metering ---------------------------------------------------------

    def _record(self, raw: bytes, timing: Timing, request_messages, server_before):
        streamed = raw.lstrip().startswith(b"data:")
        if streamed:
            usage, assistant = _usage_and_text_from_stream(raw)
        else:
            try:
                body = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                body = {}
            usage = usage_from_response(body)
            choices = body.get("choices") or [{}]
            assistant = (choices[0] or {}).get("message") or None
            # Without streaming there is no first-token boundary, so decode
            # throughput is not measurable. Recorded as absent, not estimated
            # from total time -- that would be a different metric wearing this
            # metric's name.
            timing = Timing(started_at=timing.started_at, first_token_at=None,
                            finished_at=timing.finished_at)

        server_after = None
        if self.cfg.scrape_prometheus:
            server_after = scrape(self.cfg.prometheus_url, self.cfg.prometheus_timeout_s)

        previous = self.state.swap_messages(request_messages, assistant)

        row = build_row(
            call_index=self.state.next_index(),
            usage=usage,
            timing=timing,
            previous_messages=previous,
            current_messages=request_messages,
            server_before=server_before,
            server_after=server_after,
            token_counter=self.token_counter,
            prefix_counter=self.prefix_counter,
            active_param_count=self.cfg.active_param_count,
            bytes_per_param=self.cfg.bytes_per_param(),
            peak_bandwidth_bytes_per_s=self.cfg.peak_bandwidth(),
            run_label=self.cfg.run_label,
            task_id=self.cfg.task_id,
            extra={"streamed": streamed, "upstream": self.cfg.upstream_base_url},
        )
        self.writer.write(row)


def serve(cfg: Optional[MeterConfig] = None) -> None:
    cfg = cfg or MeterConfig.from_env()
    writer = RowWriter(cfg.episode_dir())
    state = EpisodeState()

    counter = load_token_counter(cfg.tokenizer)
    prefix_counter = load_prefix_counter(cfg.tokenizer)

    handler = type("BoundHandler", (Handler,), {
        "cfg": cfg, "writer": writer, "state": state,
        "token_counter": counter, "prefix_counter": prefix_counter,
    })
    httpd = ThreadingHTTPServer((cfg.listen_host, cfg.listen_port), handler)

    log.info("meter proxy on http://%s:%d -> %s", cfg.listen_host, cfg.listen_port,
             cfg.upstream_base_url)
    log.info("rows -> %s", writer.path)
    # Both warnings fire before any GPU time is spent, so a sweep that cannot
    # produce a metric is visible up front rather than discovered in analysis.
    if cfg.mbu_inputs_missing():
        log.warning("MBU will be absent; missing: %s", ", ".join(cfg.mbu_inputs_missing()))
    if prefix_counter is None:
        log.warning(
            "no tokenizer -- weighted_prefix_efficiency will be undefined on every "
            "row (set METER_TOKENIZER=<hf repo>). Other metrics unaffected."
        )
    httpd.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[meter] %(levelname)s %(message)s")
    serve()
