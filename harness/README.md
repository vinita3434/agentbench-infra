# harness/

Placeholder for **context-management strategies**, implemented as Pi extensions.

The point of this benchmark's later rounds is to hold the *model* fixed and vary
the *harness*. Each strategy (e.g. aggressive summarization, retrieval-augmented
context, sliding window) lands here as a self-contained Pi extension, and is
selected at run time without touching model config:

```bash
run/run_task.sh --task <id> --mode native \
  --extension harness/summarize_v1.js      # -> pi ... --extension <path>
```

`run_task.sh` already threads an `--extension` flag (and `PI_EXTENSION` env)
straight through to Pi and records which extension produced each result in
`run_record.json`. Dropping in a new strategy is a config change, not a rewrite.

See Pi's extension docs: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md
