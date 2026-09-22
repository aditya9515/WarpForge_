#!/usr/bin/env python3
"""Convert a trtexec timing trace into a WarpForge benchmark record."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument(
        "--timing-scope",
        choices=("device-resident", "transfer-inclusive"),
        required=True,
    )
    parser.add_argument("--warmup-ms", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(samples),
        "minimum": min(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p95": percentile(samples, 0.95),
        "standard_deviation": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def driver_version() -> str:
    driver = ctypes.WinDLL("nvcuda.dll")
    version = ctypes.c_int()
    result = driver.cuDriverGetVersion(ctypes.byref(version))
    if result != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with status {result}")
    return f"{version.value // 1000}.{(version.value % 1000) // 10}"


def main() -> None:
    arguments = parse_arguments()
    if arguments.warmup_ms < 0 or arguments.iterations <= 0:
        raise ValueError("warmup-ms must be non-negative and iterations must be positive")
    trace = json.loads(arguments.times.read_text(encoding="utf-8"))
    validation = json.loads(arguments.validation.read_text(encoding="utf-8"))
    if len(trace) != arguments.iterations:
        raise ValueError(
            f"expected {arguments.iterations} timing samples, found {len(trace)}"
        )
    if validation["precision"] != arguments.precision:
        raise ValueError("precision does not match the validation record")
    if not validation["correctness"]["passed"]:
        raise ValueError("TensorRT validation did not pass")

    field = "computeMs" if arguments.timing_scope == "device-resident" else "latencyMs"
    samples = [float(sample[field]) for sample in trace]
    if any(not math.isfinite(sample) or sample < 0.0 for sample in samples):
        raise ValueError("timing trace contains an invalid sample")
    sample_statistics = summarize(samples)
    properties = torch.cuda.get_device_properties(0)
    shape = validation["input_shape"]
    if shape != [1, 128, 512] or validation["output_shape"] != shape:
        raise ValueError("TensorRT engine does not have the fixed Stage 10 shape")

    result = {
        "schema_version": 1,
        "metadata": {
            "operation": "miniinfer_block",
            "implementation": "tensorrt_trtexec",
            "data_type": arguments.precision,
            "timing_scope": (
                "gpu-compute-device-resident"
                if arguments.timing_scope == "device-resident"
                else "gpu-h2d-compute-d2h"
            ),
            "dimensions": {
                "batch": 1,
                "sequence": 128,
                "hidden_size": 512,
                "attention_heads": 8,
                "head_dimension": 64,
                "intermediate_size": 1536,
            },
            "launch": {
                "grid": [0, 0, 0],
                "block": [0, 0, 0],
                "dynamic_shared_memory_bytes": 0,
            },
            "gpu_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_global_memory_bytes": properties.total_memory,
            "cuda_driver_version": driver_version(),
            "cuda_runtime_version": str(torch.version.cuda),
            "build_type": "TensorRT serialized engine",
            "compiler": f"TensorRT {validation['tensorrt_version']} / trtexec",
            "git_commit": git_commit(),
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "config": {
            "warmup_duration_ms": arguments.warmup_ms,
            "measurement_iterations": arguments.iterations,
            "seed": 2027,
            "data_transfers": arguments.timing_scope == "transfer-inclusive",
            "cuda_graph": False,
            "inference_streams": 1,
        },
        "statistics_ms": sample_statistics,
        "correctness": validation["correctness"],
        "metrics": {
            "tokens_per_second_from_median": 128000.0
            / sample_statistics["median"],
            "engine_bytes": validation["engine_bytes"],
            "engine_device_memory_bytes": validation["engine_device_memory_bytes"],
            "engine_sha256": validation["engine_sha256"],
            "mean_enqueue_ms": statistics.fmean(
                float(sample["endEnqMs"]) - float(sample["startEnqMs"])
                for sample in trace
            ),
            "mean_h2d_ms": statistics.fmean(float(sample["h2dMs"]) for sample in trace),
            "mean_compute_ms": statistics.fmean(
                float(sample["computeMs"]) for sample in trace
            ),
            "mean_d2h_ms": statistics.fmean(float(sample["d2hMs"]) for sample in trace),
        },
        "samples_ms": samples,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"{arguments.output.name}: median {sample_statistics['median']:.6f} ms, "
        f"p95 {sample_statistics['p95']:.6f} ms"
    )


if __name__ == "__main__":
    main()
