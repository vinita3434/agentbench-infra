#!/usr/bin/env python3
"""End-to-end: the proxy must be transparent, and must record without breaking."""
import json
import pathlib
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from meter.config import MeterConfig  # noqa: E402
from meter.proxy import Handler, EpisodeState  # noqa: E402
from meter.record import RowWriter  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

BODY = {
    "id": "c1", "object": "chat.completion", "model": "local-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020,
              "prompt_tokens_details": {"cached_tokens": 800}},
}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        payload = json.dumps(BODY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _serve(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


@pytest.fixture
def stack():
    up = HTTPServer(("127.0.0.1", 0), Upstream)
    _serve(up)
    tmp = tempfile.mkdtemp()
    cfg = MeterConfig(
        upstream_base_url=f"http://127.0.0.1:{up.server_port}/v1",
        prometheus_url=None, scrape_prometheus=False,
        run_label="test-model", task_id="t1",
        results_dir=pathlib.Path(tmp),
        active_param_count=3.3e9, quantization="fp8", gpu="H100_SXM",
    )
    writer = RowWriter(cfg.episode_dir())
    handler = type("H", (Handler,), {
        "cfg": cfg, "writer": writer, "state": EpisodeState(), "token_counter": len,
    })
    prox = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    _serve(prox)
    yield prox, writer
    prox.shutdown()
    up.shutdown()


def _post(port, messages):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "local-model", "messages": messages}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_proxy_is_transparent(stack):
    """The caller must get exactly what upstream sent."""
    prox, _ = stack
    got = _post(prox.server_port, [{"role": "user", "content": "hi"}])
    assert got == BODY


def test_proxy_records_a_row_per_call(stack):
    prox, writer = stack
    _post(prox.server_port, [{"role": "system", "content": "sys"},
                             {"role": "user", "content": "one"}])
    _post(prox.server_port, [{"role": "system", "content": "sys"},
                             {"role": "user", "content": "one"},
                             {"role": "user", "content": "two"}])
    time.sleep(0.3)

    rows = [json.loads(l) for l in writer.path.read_text().splitlines()]
    assert len(rows) == 2
    assert [r["call_index"] for r in rows] == [1, 2]

    first, second = rows
    # Turn 1: nothing was reusable -> efficiency undefined, NOT 0.
    assert first["prefix_reusable_tokens"] == 0
    assert first["weighted_prefix_efficiency"] is None
    assert first["weighted_prefix_efficiency_reason"] == "undefined:zero_denominator"
    # Per-request cache rate is a real number on both rows.
    assert first["prefix_cache_rate"] == pytest.approx(0.8)
    # Turn 2 saw turn 1's context, so some prefix was eligible.
    assert second["prefix_matched_messages"] >= 2


def test_non_streamed_has_no_decode_rate(stack):
    """This mock does not stream, so decode throughput must be absent."""
    prox, writer = stack
    _post(prox.server_port, [{"role": "user", "content": "hi"}])
    time.sleep(0.3)
    row = json.loads(writer.path.read_text().splitlines()[0])
    assert row["decode_tokens_per_s"] is None
    assert row["decode_tokens_per_s_reason"] == "undefined:not_streamed"
    assert row["mbu"] is None       # depends on decode rate


def test_metering_failure_does_not_break_the_call(stack):
    """A broken writer must not surface to the caller."""
    prox, writer = stack
    writer.path = pathlib.Path("/nonexistent-dir/rows.jsonl")  # writes will fail
    got = _post(prox.server_port, [{"role": "user", "content": "hi"}])
    assert got == BODY          # episode continues regardless
