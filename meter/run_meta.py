#!/usr/bin/env python3
"""Write the run-level record that makes a trajectory interpretable later.

The per-turn rows are only half the artifact. Comparing serving configs weeks
from now means answering "what exactly produced these numbers" -- which weights,
which quantization, which SGLang build, which launch flags, which GPU, which
harness commit. None of that is recoverable from the rows themselves, and none
of it is reliably recoverable from memory.

Everything here is read-only with respect to the existing tree: serve/models.yaml
and results/serve/*.json are read if present, never written. Output goes to the
metering path's own results location.

Missing stays missing here too. If nvidia-smi is absent, the GPU block is null
with a reason -- not "unknown", not a guess from the config.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Optional

from .config import MeterConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_META_FILENAME = "run_meta.json"


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def gpu_info() -> Dict[str, Any]:
    """GPU identity and memory, from nvidia-smi. Absent on a machine without it."""
    query = "name,memory.total,driver_version,compute_cap"
    raw = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    if not raw:
        return {"available": False, "reason": "nvidia-smi unavailable or failed"}
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "name": parts[0],
                "memory_total": parts[1],
                "driver_version": parts[2],
                "compute_capability": parts[3],
            })
    return {"available": True, "count": len(gpus), "gpus": gpus}


def sglang_version() -> Optional[str]:
    return _run(["python3", "-c", "import sglang; print(sglang.__version__)"])


def harness_sha() -> Optional[str]:
    return _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])


def harness_dirty() -> Optional[bool]:
    """Whether the working tree had uncommitted changes at run time.

    A SHA alone is misleading if the tree was dirty -- the recorded commit would
    not reproduce the run.
    """
    out = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    if out is None:
        return None
    return bool(out.strip())


def registry_entry(model_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """The serve/models.yaml entry for this model, read-only."""
    if not model_key:
        return None
    registry = REPO_ROOT / "serve" / "models.yaml"
    if not registry.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(registry.read_text()) or {}
        return (data.get("models") or {}).get(model_key)
    except Exception:
        return None


def launch_record(model_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Newest serve/launch.sh record for this model: exact flags it served with.

    This is the authoritative source for launch flags -- it is written by the
    process that actually started the server, so it cannot drift from reality
    the way a hand-copied flag list would.
    """
    if not model_key:
        return None
    serve_dir = REPO_ROOT / "results" / "serve"
    if not serve_dir.exists():
        return None
    candidates = sorted(serve_dir.glob(f"{model_key}-*.json"), reverse=True)
    for path in candidates:
        try:
            return {"path": str(path), "record": json.loads(path.read_text())}
        except Exception:
            continue
    return None


def build_run_meta(cfg: MeterConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entry = registry_entry(cfg.run_label)
    meta: Dict[str, Any] = {
        "schema": "meter.run_meta.v1",
        "written_at_unix": time.time(),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_label": cfg.run_label,
        "task_id": cfg.task_id,
        "model": {
            "registry_key": cfg.run_label,
            "hf_repo": (entry or {}).get("hf_repo"),
            "quantization": cfg.quantization or (entry or {}).get("quantization"),
            "tool_parser": (entry or {}).get("parser"),
            "reasoning_parser": (entry or {}).get("reasoning_parser"),
            "context_length": (entry or {}).get("context_length"),
            "tp_size": (entry or {}).get("tp_size"),
            "active_param_count": cfg.active_param_count,
            "total_param_count": cfg.total_param_count,
        },
        "serving": {
            "sglang_version_installed": sglang_version(),
            "sglang_version_pinned": (entry or {}).get("sglang_version"),
            "launch_record": launch_record(cfg.run_label),
            "upstream_base_url": cfg.upstream_base_url,
            "prometheus_url": cfg.prometheus_url,
        },
        "hardware": {
            "gpu_config_key": cfg.gpu,
            "peak_bandwidth_bytes_per_s": cfg.peak_bandwidth(),
            "detected": gpu_info(),
        },
        "harness": {
            "git_sha": harness_sha(),
            "git_dirty": harness_dirty(),
            "repo_root": str(REPO_ROOT),
        },
        "meter_config": cfg.as_dict(),
        # Stated up front so a sweep that cannot produce MBU is visible before
        # the GPU time is spent, not discovered in the analysis.
        "mbu_inputs_missing": cfg.mbu_inputs_missing(),
    }
    if extra:
        meta["extra"] = extra
    return meta


def write_run_meta(cfg: MeterConfig, extra: Optional[Dict[str, Any]] = None) -> pathlib.Path:
    directory = cfg.episode_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_META_FILENAME
    path.write_text(json.dumps(build_run_meta(cfg, extra), indent=2, default=str))
    return path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="[meter] %(levelname)s %(message)s")
    written = write_run_meta(MeterConfig.from_env())
    print(f"[meter] run meta -> {written}")
