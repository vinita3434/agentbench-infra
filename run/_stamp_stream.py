#!/usr/bin/env python3
"""Timestamp Pi's --mode json event stream at capture time.

Pi's JSON events carry no timestamps, so the only way to measure TTFT / per-turn
latency in Mode A is to record when each event line *arrives*. This reads Pi's
stdout line by line, passes each raw line straight through to our stdout (so a
`> pi_log.jsonl` redirect still yields the untouched raw log), and appends a
timestamped record to the file named in argv[1]:

    {"_t": <seconds since first byte>, "event": <parsed event or raw>}

Run under `python3 -u` so lines are not buffered and timings stay honest.
"""
import json
import sys
import time


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: _stamp_stream.py <events.timed.jsonl>")
    timed = open(sys.argv[1], "w")
    t0 = None
    for raw in iter(sys.stdin.readline, ""):
        # pass raw through immediately
        sys.stdout.write(raw)
        sys.stdout.flush()
        line = raw.strip()
        if not line:
            continue
        now = time.monotonic()
        if t0 is None:
            t0 = now
        t = round(now - t0, 4)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "_raw", "raw": line}
        timed.write(json.dumps({"_t": t, "event": event}) + "\n")
        timed.flush()
    timed.close()


if __name__ == "__main__":
    main()
