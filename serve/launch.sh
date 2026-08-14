#!/usr/bin/env bash
#
# launch.sh --model <key>
#
# Serve <key> on 0.0.0.0:30000 as an OpenAI-compatible endpoint, prove tool
# calling actually works, and record exactly how it was launched.
#
#   1. Read model settings from serve/models.yaml.
#   2. Verify the installed SGLang version matches the pin -- hard error if not.
#   3. Start SGLang with the correct quantization / parser / context / TP.
#   4. Wait until the server reports ready.
#   5. Live tool-call test: POST a request with a tool definition and assert a
#      real tool_calls array comes back. A mismatched parser is a SILENT failure
#      -- catch it here, not three hours into a sweep. Prints PASS/FAIL.
#   6. Write a timestamped run record to results/serve/ with the model key,
#      SGLang version, and every launch flag used.
#
# serve/ knows nothing about Pi. It serves an endpoint; that's all.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
REGISTRY="$HERE/_registry.py"
RESULTS_DIR="$REPO_ROOT/results/serve"

MODEL=""
HOST="0.0.0.0"
PORT="30000"
SERVED_NAME="local-model"   # stable name so the harness config never changes
                            # when you swap the model behind it.
KEEP_FG=0                   # --foreground: keep server attached (default: detach)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --served-name) SERVED_NAME="$2"; shift 2 ;;
    --foreground) KEEP_FG=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "usage: launch.sh --model <key> [--port N] [--foreground]" >&2; exit 2; }

# --- read config ---------------------------------------------------------
HF_REPO="$(python3 "$REGISTRY" "$MODEL" hf_repo)"
WANT_VERSION="$(python3 "$REGISTRY" "$MODEL" sglang_version)"
QUANT="$(python3 "$REGISTRY" "$MODEL" quantization)"
PARSER="$(python3 "$REGISTRY" "$MODEL" parser)"
REASONING_PARSER="$(python3 "$REGISTRY" "$MODEL" reasoning_parser)"
CTX="$(python3 "$REGISTRY" "$MODEL" context_length)"
TP="$(python3 "$REGISTRY" "$MODEL" tp_size)"

# --- verify installed SGLang matches the pin -----------------------------
CUR_VERSION="$(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || true)"
if [[ -z "$CUR_VERSION" ]]; then
  echo "[launch] SGLang is not installed. Run: serve/setup.sh --model $MODEL" >&2
  exit 1
fi
if [[ "$CUR_VERSION" != "$WANT_VERSION" ]]; then
  echo "[launch] SGLang version mismatch: installed $CUR_VERSION, model '$MODEL' needs $WANT_VERSION." >&2
  echo "[launch] Run: serve/setup.sh --model $MODEL" >&2
  exit 1
fi

# --- assemble launch flags (record these verbatim) -----------------------
FLAGS=(
  --model-path "$HF_REPO"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --quantization "$QUANT"
  --tool-call-parser "$PARSER"
  --context-length "$CTX"
  --tp-size "$TP"
)
[[ -n "$REASONING_PARSER" ]] && FLAGS+=( --reasoning-parser "$REASONING_PARSER" )

echo "[launch] model=$MODEL repo=$HF_REPO sglang=$CUR_VERSION"
echo "[launch] flags: python -m sglang.launch_server ${FLAGS[*]}"

mkdir -p "$RESULTS_DIR"
SERVER_LOG="$RESULTS_DIR/${MODEL}.server.log"

# --- start server (background) -------------------------------------------
python -m sglang.launch_server "${FLAGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[launch] server pid=$SERVER_PID  log=$SERVER_LOG"

cleanup_fail() {
  echo "[launch] shutting server down (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
}

# --- wait for ready ------------------------------------------------------
BASE="http://127.0.0.1:${PORT}"
echo -n "[launch] waiting for ready "
READY=0
for _ in $(seq 1 600); do   # up to ~10 min for cold weight load
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "[launch] server process died during startup. Tail of log:" >&2
    tail -n 40 "$SERVER_LOG" >&2
    exit 1
  fi
  if curl -sf "$BASE/health_generate" >/dev/null 2>&1; then READY=1; break; fi
  echo -n "."
  sleep 1
done
echo ""
if [[ "$READY" -ne 1 ]]; then
  echo "[launch] server did not become ready in time." >&2
  tail -n 40 "$SERVER_LOG" >&2
  cleanup_fail
  exit 1
fi
echo "[launch] server READY at $BASE"

# --- live tool-call test -------------------------------------------------
# Ask the model to check weather with a tool available; a correct parser
# returns choices[0].message.tool_calls as a non-empty array.
echo "[launch] running tool-call test..."
TEST_RESP="$(curl -s "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @- <<JSON
{
  "model": "$SERVED_NAME",
  "messages": [
    {"role": "user", "content": "What is the weather in San Francisco? Use the tool."}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "temperature": 0,
  "max_tokens": 256
}
JSON
)"

TOOLCALL_TEST="FAIL"
if python3 - "$TEST_RESP" <<'PY'
import json, sys
try:
    resp = json.loads(sys.argv[1])
    tcs = resp["choices"][0]["message"].get("tool_calls") or []
    ok = isinstance(tcs, list) and len(tcs) > 0 and tcs[0]["function"]["name"] == "get_weather"
    if ok:
        print("[launch]   -> got tool_call:", json.dumps(tcs[0]["function"]))
    sys.exit(0 if ok else 1)
except Exception as e:
    print("[launch]   -> could not parse tool_calls:", e)
    sys.exit(1)
PY
then
  TOOLCALL_TEST="PASS"
fi

if [[ "$TOOLCALL_TEST" == "PASS" ]]; then
  echo "[launch] TOOL-CALL TEST: PASS"
else
  echo "[launch] TOOL-CALL TEST: FAIL" >&2
  echo "[launch] Parser '$PARSER' likely wrong for $MODEL / SGLang $CUR_VERSION." >&2
  echo "[launch] Raw response (truncated):" >&2
  echo "${TEST_RESP:0:800}" >&2
fi

# --- write run record ----------------------------------------------------
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RECORD="$RESULTS_DIR/${MODEL}-${TS}.json"
python3 - "$RECORD" "$MODEL" "$HF_REPO" "$CUR_VERSION" "$SERVED_NAME" \
                    "$HOST" "$PORT" "$QUANT" "$PARSER" "$REASONING_PARSER" \
                    "$CTX" "$TP" "$TOOLCALL_TEST" "$TS" <<'PY'
import json, sys
(path, model, hf_repo, sglang_version, served, host, port, quant, parser,
 reasoning, ctx, tp, toolcall, ts) = sys.argv[1:15]
flags = [
    "--model-path", hf_repo, "--served-model-name", served,
    "--host", host, "--port", port, "--quantization", quant,
    "--tool-call-parser", parser, "--context-length", ctx, "--tp-size", tp,
]
if reasoning:
    flags += ["--reasoning-parser", reasoning]
record = {
    "timestamp_utc": ts,
    "model_key": model,
    "hf_repo": hf_repo,
    "sglang_version": sglang_version,
    "served_model_name": served,
    "endpoint": f"http://{host}:{port}/v1",
    "quantization": quant,
    "tool_call_parser": parser,
    "reasoning_parser": reasoning or None,
    "context_length": int(ctx),
    "tp_size": int(tp),
    "launch_flags": flags,
    "tool_call_test": toolcall,
}
with open(path, "w") as f:
    json.dump(record, f, indent=2)
print(f"[launch] run record -> {path}")
PY

# --- exit behavior -------------------------------------------------------
if [[ "$TOOLCALL_TEST" != "PASS" ]]; then
  echo "[launch] Tool-call test failed -- NOT leaving a broken server up." >&2
  cleanup_fail
  exit 1
fi

if [[ "$KEEP_FG" -eq 1 ]]; then
  echo "[launch] serving in foreground (Ctrl-C to stop). Endpoint: $BASE/v1"
  wait "$SERVER_PID"
else
  echo "[launch] server running (pid $SERVER_PID). Endpoint: $BASE/v1"
  echo "[launch] stop with: kill $SERVER_PID"
fi
