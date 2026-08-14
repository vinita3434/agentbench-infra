#!/usr/bin/env python3
"""Can a task actually be *scored*, on the machine we run on today?

`fetch_tasks.py --simplest` answers "is this task easy for the agent". That is
only half the selection. A task is useless as a benchmark item if the harness
cannot turn a patch into a pass/fail, and two things break that:

  1. **No eval image.** Verification runs inside SWE-PolyBench's pre-built
     per-instance image on GHCR. Only a subset of the dataset has one published
     (~153 of the 550 single-function tasks). Checked live, over the registry
     API -- no docker pull, so it costs a few hundred ms per task.

  2. **The test process never produces parseable results.** Established
     empirically by grading the *gold* patch (`verify_task.sh --gold`): if the
     dataset's own patch cannot be scored, no model's patch can be either.
     Those runs live in `results/_gold/` and are what STATUS below encodes.

The failures cluster by *family* (repo + test framework), not by task -- every
TensorFlow-backed suite dies the same way under x86 emulation on Apple silicon
("Illegal instruction" = SIGILL) -- so one gold run per family generalizes.

Statuses:
  PROVEN     a gold run in results/_gold/ came back resolved=True
  PLAUSIBLE  same framework/runtime shape as a PROVEN family, not yet graded
  BROKEN     a gold run failed to produce parseable results (Mode A / Mac)
  UNPROVEN   no gold evidence either way

BROKEN is a statement about **Mode A on this Mac**, not about the task. The
SIGILL crashes come from emulating x86 AVX on arm64; on the RunPod box (Mode B,
native x86) they are expected to disappear, at which point those families get
re-graded and promoted. Hence `tier2_reason` rather than silent exclusion.
"""
import concurrent.futures
import json
import ssl
import urllib.error
import urllib.request

import certifi

REGISTRY = "ghcr.io"
IMAGE_PREFIX = "timesler/swe-polybench.eval.x86_64."
ACCEPT = ",".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])

PROVEN, PLAUSIBLE, BROKEN, UNPROVEN = "proven", "plausible", "broken", "unproven"

# repo -> (status, note). Keyed by repo because the test runtime is a property
# of the repo's suite, not of the individual task.
STATUS = {
    # --- graded green: gold patch scored resolved=True -----------------------
    "prettier/prettier":       (PROVEN, "jest --json on stdout"),
    "serverless/serverless":   (PROVEN, "mocha"),
    "sveltejs/svelte":         (PROVEN, "jest via nvm"),
    "langchain-ai/langchain":  (PROVEN, "pytest, pure-python (no TF/torch)"),
    # --- same shape as a proven family, not yet graded ----------------------
    "mui/material-ui":         (PLAUSIBLE, "mocha, same runner as serverless"),
    "coder/code-server":       (PLAUSIBLE, "jest, same runner as prettier"),
    # --- graded red under Mode A (Mac, x86 emulation) -----------------------
    "keras-team/keras":        (BROKEN, "TensorFlow SIGILL under x86 emulation"),
    "huggingface/transformers": (BROKEN, "torch/TF SIGILL under x86 emulation"),
    "yt-dlp/yt-dlp":           (BROKEN, "test process produced no parseable results"),
    "mrdoob/three.js":         (BROKEN, "headless GL; no parseable results"),
    "google/gson":             (BROKEN, "maven surefire XML not produced"),
    # --- Java: gold runs never completed at all -----------------------------
    "apache/dubbo":            (UNPROVEN, "maven; gold run produced no verify.json"),
    "apache/rocketmq":         (UNPROVEN, "maven; gold run produced no verify.json"),
    "trinodb/trino":           (UNPROVEN, "maven; gold run produced no verify.json"),
    # --- heavyweight desktop suite ------------------------------------------
    "microsoft/vscode":        (UNPROVEN, "electron + xvfb + full yarn compile"),
}

# Statuses admitted by `--verifiable`.
RUNNABLE = (PROVEN, PLAUSIBLE)


def status(repo):
    """(status, note) for a repo; unknown repos are UNPROVEN, not excluded silently."""
    return STATUS.get(repo, (UNPROVEN, "no gold evidence for this repo"))


def framework(test_command):
    """Coarse test-runner label, used for spreading gold-verification samples."""
    tc = test_command or ""
    # Order matters: several suites pass a --json-ish reporter flag, so the
    # runner name has to be matched before the output-format flag.
    if "surefire" in tc:
        return "maven"
    if "pytest" in tc:
        return "pytest"
    if "mocha" in tc:
        return "mocha"
    if "xvfb" in tc:
        return "xvfb-electron"
    if "jest" in tc:
        return "jest"
    if "npm run test" in tc or "yarn test" in tc:
        # Repo's own `test` script; the runner underneath is whatever the repo
        # pinned (svelte -> mocha, code-server -> jest). What the parser cares
        # about is that it emits a JSON reporter, which these all do.
        return "npm-script"
    if "--json" in tc:
        return "jest"
    return "unknown"


def _ctx():
    return ssl.create_default_context(cafile=certifi.where())


def has_eval_image(instance_id, tag="latest", timeout=30):
    """True if the per-instance eval image is published and anonymously pullable.

    GHCR answers 403/DENIED (not 404) for a repository an anonymous caller
    cannot see, so anything other than a 200 on the manifest means "we cannot
    verify this task", which is the only distinction that matters here.
    """
    repo = f"{IMAGE_PREFIX}{instance_id}"
    ctx = _ctx()
    try:
        tok_url = (f"https://{REGISTRY}/token?scope=repository:{repo}:pull"
                   f"&service={REGISTRY}")
        token = json.load(urllib.request.urlopen(tok_url, context=ctx,
                                                 timeout=timeout))["token"]
        req = urllib.request.Request(
            f"https://{REGISTRY}/v2/{repo}/manifests/{tag}",
            method="HEAD",
            headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
        )
        return urllib.request.urlopen(req, context=ctx, timeout=timeout).status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError,
            json.JSONDecodeError, TimeoutError):
        return False


def image_map(instance_ids, workers=12, tag="latest"):
    """{instance_id: bool} for many ids at once. Keep `workers` modest --
    GHCR's anonymous token endpoint starts refusing above ~24 in flight."""
    ids = list(instance_ids)
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        return dict(zip(ids, ex.map(lambda i: has_eval_image(i, tag=tag), ids)))
