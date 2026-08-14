# agent-serving-bench

An evaluation harness for benchmarking open-weight LLMs on **agentic coding
tasks**. It runs real coding-agent trajectories (the [Pi coding
agent](https://github.com/earendil-works/pi), unmodified) against
[SWE-PolyBench](https://huggingface.co/datasets/AmazonScience/SWE-PolyBench)
tasks, holds the harness fixed, and measures how different models perform. Later
rounds vary the harness instead.

Pi speaks OpenAI-compatible JSON, so swapping the model behind it is a **config
change, not a code change**.

## Metrics

Resolve rate, cost per verified task, TTFT across turn index, KV cache hit rate,
VRAM utilization.

## Two deployment modes

| | Mode A (testing, now) | Mode B (real runs, later) |
|---|---|---|
| Where | Docker on Mac (x86 emulated) | Single RunPod box |
| Model | frontier via OpenRouter | SGLang on `localhost:30000` |
| Pi runs | inside a container | native, on the pod filesystem |
| Isolation | container teardown | `git reset --hard && git clean -fdx` |

`run/run_task.sh` supports both with one flag (`--mode docker|native`) — same
logic, no restructuring. RunPod can't do Docker-in-Docker, hence native there.

## Layout

```
serve/      runs on the GPU box; serves an OpenAI-compatible endpoint (knows nothing about Pi)
  models.yaml   central model registry — add a model = add one entry
  setup.sh      pin/install exact SGLang version + HF auth
  download.sh   fetch weights from HuggingFace
  launch.sh     serve on :30000, live tool-call test, write run record
tasks/
  fetch_tasks.py   stratified SWE-PolyBench subset -> tasks/data/<id>.json
  verifiability.py which tasks can actually be *scored* (gold evidence + GHCR image)
run/
  pi_config/       Pi provider configs (kept separate from harness config)
    models.openrouter.json    Mode A: OpenRouter
    models.selfhosted.json    Mode B: SGLang on :30000 (served as "local-model")
  run_task.sh      the core loop (docker or native); captures a timestamped event stream
  verify_task.sh   score a patch vs hidden F2P/P2P tests (SWE-PolyBench GHCR image)
  verify_sample.sh gold-verify one task per (repo, framework) -- proves the set
  extract_metrics.py  per-run trajectory metrics (turns, TTFT, per-turn reasoning/actions)
  report.py        consolidate metrics + verify -> terminal table + summary.json + report.html
  parse_results.py lightweight JSON-log summary table (original)
harness/    future: context strategies as Pi extensions (see harness/README.md)
results/    per-run outputs + config records
Dockerfile.test   Mac-only, Mode A
```

## Design constraints honored

- **Model config and harness config are separate.** `serve/models.yaml` is the
  model registry; `run/pi_config/` + `harness/` are the harness. Later rounds
  vary the harness while holding the model fixed — the inverse of now.
- **`serve/` knows nothing about Pi.** It serves an OpenAI-compatible endpoint,
  full stop. Swap the harness without touching those scripts.
- **Every run writes a config record.** `results/<model>/<task>/run_record.json`
  (model, provider, mode, base commit, all Pi flags, extension, harness git SHA)
  and `results/serve/<model>-<ts>.json` (SGLang version + every launch flag).
  Any result is reconstructable.
- **Context strategies are swappable.** `harness/` holds Pi extensions;
  `run_task.sh --extension <path>` threads one through and records it. Dropping
  one in is a config change.

---

## Quickstart — Mode A, prove one task end to end (today's goal)

Prereqs: Docker running, an OpenRouter API key.

```bash
# 0. Python tooling for fetch/parse (a venv is fine)
pip install -r requirements.txt

# 1. Pull the working set: easiest tasks that can actually be scored
python tasks/fetch_tasks.py --n 50 --simplest --verifiable --clean

# 2. Pick a task id (any file under tasks/data/)
ls tasks/data/*.json

# 3. Run it through Pi against a frontier model, in a container
export OPENROUTER_API_KEY=sk-or-...
run/run_task.sh --task <instance_id> --mode docker

# 4. Score it against the hidden F2P/P2P tests (SWE-PolyBench GHCR image)
run/verify_task.sh --task <instance_id>

# 5. Consolidated report (terminal table + summary.json + report.html)
python run/report.py --open
```

Outputs land in `results/<model>/<instance_id>/`:
`candidate.patch`, `pi_log.jsonl`, `events.timed.jsonl` (timestamped stream),
`pi_stderr.log`, `run_record.json`, `metrics.json`, `verify.json`.

### Task selection — easy ∩ verifiable

Round one wants tasks a model has a real chance of solving **and** that the
harness can actually score. Those are two independent filters and the second is
by far the tighter one:

| gate | flag | 2110 dataset rows → |
|---|---|---|
| single-function, ≤1500-char patch, 1–5 F2P | `--simplest` | **550** |
| repo's test suite has graded green (or matches one that has) | `--verifiable` | 428 |
| per-instance eval image published on GHCR | `--verifiable` | **86** |

(Only 153 of the 550 have an eval image at all; 86 of those are also in a
gradeable family. So the 50 are drawn from a pool of 86, not from 2110.)

`tasks/verifiability.py` holds the second gate. Its repo table is *evidence*,
not opinion: a family is green only because `verify_task.sh --gold` scored the
dataset's own patch `resolved=True` for it. If gold can't be scored, no model's
patch can be either — that task would report a 0 that means "harness broke",
not "model failed", which is the one failure mode that silently corrupts a
resolve rate.

The current set (`--n 50 --simplest --verifiable`) is 30 JavaScript, 12
TypeScript, 8 Python across serverless (13), mui (11), svelte (9), langchain
(8), prettier (8), code-server (1); 43 Bug Fix, 6 Feature, 1 Refactoring.

**Java is absent and Python is thin, deliberately.** Under Mode A the keras /
transformers / yt-dlp suites die with `Illegal instruction` (SIGILL) — x86 AVX
emulated on arm64 — and no Java gold run has ever produced a surefire XML.
Those ~30 image-backed tasks are tracked as `broken` / `unproven` in
`verifiability.py`, not deleted: Mode B runs native x86 on the RunPod box,
where the SIGILL class of failure should disappear. Re-grade with
`run/verify_sample.sh --only apache/dubbo` there and promote what passes.

**Preflight a task before paying for it** (`--gold-first`). Grading the gold
patch is model-independent, so if gold cannot be scored, no agent run on that
task can produce a result either — it will report ERROR no matter how good the
model is. `--gold-first` checks that up front, reuses a cached
`results/_gold/<task>/verify.json` when one exists, and exits **3** without
running the agent when the task is unscoreable here:

```bash
run/run_task.sh --task <id> --mode docker --gold-first
```

Real case: `langchain-5584` spent a full agent run and a 26-minute image pull
before failing with `Illegal instruction` — ChromaDB's `hnswlib` is compiled
with AVX, which x86-on-arm64 emulation does not implement. The preflight would
have caught that for the price of the pull alone, and the pull is kept for the
verification that follows.

Proving a set costs a handful of runs, not 50, because failures cluster by
family rather than by task:

```bash
run/verify_sample.sh --dry-run --per-family 2   # show what would be graded
run/verify_sample.sh --per-family 2             # grade gold for each family
```

### Metrics & verification

- **Trajectory metrics** (`run/extract_metrics.py`, folded into `report.py`):
  submitted, turns, turns-to-submit, per-turn reasoning + tool actions, TTFT
  trajectory, tokens (when Pi surfaces usage), wall time. TTFT comes from
  timestamping Pi's event stream at capture time (`run/_stamp_stream.py`) —
  Pi's JSON events carry no timestamps of their own.
- **Verification** (`run/verify_task.sh`): pulls SWE-PolyBench's pre-built
  per-instance image `ghcr.io/timesler/swe-polybench.eval.x86_64.<id>:latest`,
  applies the candidate patch + the `test_patch`, runs the task's `test_command`
  (jest `--json`), and reports **per-test** F2P/P2P pass/fail and %. **Resolved**
  = all F2P pass AND all P2P pass. Run `--gold` to grade the dataset's gold
  patch instead (no API key needed) — a self-check that the verifier is correct.
- **F2P** = fail-to-pass (tests tied to the fix; passing them = solved).
  **P2P** = pass-to-pass (regression guard). The agent never sees either — they
  arrive via `test_patch` after it finishes. So "all F2P pass but a P2P
  regressed" is surfaced distinctly: solved the task but broke something.
- **`report.py`** writes `results/summary.json` (rollup: resolve rate, median
  TTFT/turns, per-task rows) and `results/report.html` (KPIs, per-turn cards,
  TTFT chart).

Override the model/provider via env or flags (default:
`anthropic/claude-sonnet-5` on `openrouter`):

```bash
MODEL=anthropic/claude-opus-4.8 run/run_task.sh --task <id> --mode docker
```

## Mode B — real runs on RunPod (later)

```bash
# on the pod
serve/setup.sh    --model qwen3-coder-30b     # pin + install exact SGLang, HF auth
serve/download.sh --model qwen3-coder-30b     # fetch weights
serve/launch.sh   --model qwen3-coder-30b     # serve :30000 + tool-call PASS/FAIL + record

# point Pi at the local endpoint (served as "local-model" so this never
# changes when you swap the model behind it)
cp run/pi_config/models.selfhosted.json ~/.pi/agent/models.json

# run a task natively (isolation = git reset/clean, no Docker)
MODEL=local-model PROVIDER=selfhosted \
  run/run_task.sh --task <instance_id> --mode native --workdir /path/to/repo/checkout
```

Switching to a model that pins a different SGLang version: rerun `serve/setup.sh
--model <key>` — it detects the mismatch and **asks before reinstalling**.

## Model registry (round one)

| key | HF repo | tool parser | notes |
|---|---|---|---|
| `qwen3-coder-30b` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `qwen3_coder` | MoE, coding-tuned, GQA |
| `qwen3-32b` | `Qwen/Qwen3-32B-FP8` | `qwen25` (+reasoning `qwen3`) | Dense, GQA |
| `devstral-24b` | `mistralai/Devstral-Small-2507` | `mistral` | Dense, agent-tuned, GQA |
| `glm-4.5-air` | `zai-org/GLM-4.5-Air-FP8` | `glm45` (+reasoning `glm45`) | MoE, MLA |
| `deepseek-coder-lite` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | `deepseekv3` | MoE, MLA, small |

Parser/version pairings are explicit in `serve/models.yaml` and validated by
`launch.sh`'s live tool-call test — a mismatched parser is a silent failure, so
it's caught at launch, not hours into a sweep.

## A note on `serve/`'s three scripts

The brief's spec text described the *launch* behavior (start SGLang → tool-call
test → run record) under the `download` heading. This repo splits the three
scripts cleanly: `setup.sh` = SGLang version management + HF auth, `download.sh`
= fetch weights, `launch.sh` = serve + test + record.
