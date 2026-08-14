#!/usr/bin/env python3
"""Configuration for the metering path. Own config, no shared state.

Every constant MBU depends on lives here and is explicit. Nothing is inferred
from the model name: a lookup table that silently returns the wrong active
parameter count yields an MBU that is precise, plausible and false. Absent
config produces an absent metric, which is recoverable; a wrong constant
produces a wrong plot, which is not.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "serving"

# Peak HBM bandwidth, bytes/s. Vendor figures; the row records which was used so
# a later reader can recompute if they disagree with the number.
GPU_PEAK_BANDWIDTH = {
    "H100_SXM": 3.35e12,
    "H100_PCIE": 2.0e12,
    "A100_SXM_80GB": 2.039e12,
    "A100_PCIE_80GB": 1.935e12,
    "L40S": 0.864e12,
}

# Bytes per parameter by quantization, for the weight-read term in MBU.
BYTES_PER_PARAM = {
    "fp8": 1.0,
    "int8": 1.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "int4": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
}


@dataclass
class MeterConfig:
    """Everything the metering path needs, in one explicit object."""

    # --- where to point ---------------------------------------------------
    upstream_base_url: str = "http://localhost:30000/v1"
    prometheus_url: Optional[str] = "http://localhost:30000/metrics"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8100

    # --- run identity -----------------------------------------------------
    run_label: Optional[str] = None      # serve/models.yaml key; names the run dir
    task_id: Optional[str] = None        # set per episode by the caller

    # --- MBU constants: no defaults, deliberately -------------------------
    # Active (not total) parameters for MoE models -- only routed experts are
    # read per decoded token.
    active_param_count: Optional[float] = None
    total_param_count: Optional[float] = None
    quantization: Optional[str] = None   # key into BYTES_PER_PARAM
    gpu: Optional[str] = None            # key into GPU_PEAK_BANDWIDTH
    peak_bandwidth_bytes_per_s: Optional[float] = None  # explicit override

    # --- collection behaviour --------------------------------------------
    # HF repo or local path for the served model's tokenizer. Without it the
    # reusable-prefix denominator cannot be counted exactly, so weighted prefix
    # efficiency is reported absent rather than approximated.
    tokenizer: Optional[str] = None
    scrape_prometheus: bool = True
    prometheus_timeout_s: float = 2.0
    results_dir: pathlib.Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)

    def bytes_per_param(self) -> Optional[float]:
        if self.quantization is None:
            return None
        return BYTES_PER_PARAM.get(self.quantization.lower())

    def peak_bandwidth(self) -> Optional[float]:
        """Explicit override wins; otherwise the GPU table; otherwise None."""
        if self.peak_bandwidth_bytes_per_s is not None:
            return self.peak_bandwidth_bytes_per_s
        if self.gpu is None:
            return None
        return GPU_PEAK_BANDWIDTH.get(self.gpu)

    def mbu_inputs_missing(self) -> list:
        """Which MBU inputs are absent, for the run-meta record.

        Surfaced up front so a sweep that will produce no MBU is visible before
        it runs, not after the GPU time is spent.
        """
        missing = []
        if self.active_param_count is None:
            missing.append("active_param_count")
        if self.bytes_per_param() is None:
            missing.append("quantization->bytes_per_param")
        if self.peak_bandwidth() is None:
            missing.append("gpu->peak_bandwidth")
        return missing

    def efficiency_inputs_missing(self) -> list:
        """Whether weighted prefix efficiency can be computed at all.

        Separate from MBU because it fails for a different reason and is the
        metric most worth knowing about before a sweep starts.
        """
        return [] if self.tokenizer else ["tokenizer"]

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results_dir"] = str(self.results_dir)
        d["resolved_bytes_per_param"] = self.bytes_per_param()
        d["resolved_peak_bandwidth_bytes_per_s"] = self.peak_bandwidth()
        d["mbu_inputs_missing"] = self.mbu_inputs_missing()
        return d

    # --- loading ----------------------------------------------------------

    @classmethod
    def from_env(cls, **overrides: Any) -> "MeterConfig":
        """Build from METER_* environment variables, then explicit overrides.

        Env is the pod-friendly path: the proxy runs as its own process, so its
        configuration cannot come from the caller's function arguments.
        """
        def _f(name: str) -> Optional[float]:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return None
            try:
                return float(raw)
            except ValueError:
                return None

        cfg = cls(
            upstream_base_url=os.environ.get("METER_UPSTREAM", cls.upstream_base_url),
            prometheus_url=os.environ.get("METER_PROMETHEUS", cls.prometheus_url),
            listen_host=os.environ.get("METER_HOST", cls.listen_host),
            listen_port=int(os.environ.get("METER_PORT", cls.listen_port)),
            run_label=os.environ.get("METER_RUN_LABEL"),
            task_id=os.environ.get("METER_TASK_ID"),
            active_param_count=_f("METER_ACTIVE_PARAMS"),
            total_param_count=_f("METER_TOTAL_PARAMS"),
            quantization=os.environ.get("METER_QUANTIZATION"),
            gpu=os.environ.get("METER_GPU"),
            tokenizer=os.environ.get("METER_TOKENIZER"),
            peak_bandwidth_bytes_per_s=_f("METER_PEAK_BW"),
        )
        if os.environ.get("METER_RESULTS_DIR"):
            cfg.results_dir = pathlib.Path(os.environ["METER_RESULTS_DIR"])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_file(cls, path: os.PathLike) -> "MeterConfig":
        data = json.loads(pathlib.Path(path).read_text())
        if "results_dir" in data:
            data["results_dir"] = pathlib.Path(data["results_dir"])
        return cls(**data)

    def episode_dir(self) -> pathlib.Path:
        """results/serving/<run_label>/<task_id>/ -- own results location.

        Mirrors the shape of the existing results tree without writing into it:
        nothing here touches results/<model>/<task>/, which the frontier path
        owns.
        """
        label = self.run_label or "unlabeled"
        task = self.task_id or "unknown-task"
        return pathlib.Path(self.results_dir) / label / task
