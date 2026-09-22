#!/usr/bin/env python3
"""Validate and benchmark the Stage 9 MiniInfer block with PyTorch CUDA."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from miniinfer_model import (
    INTERMEDIATE_NAMES,
    WEIGHT_NAMES,
    MiniInferBlock,
    MiniInferConfig,
    load_fixture,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("benchmarks/fixtures/miniinfer_full"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/stage10/pytorch"),
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def tolerance(name: str, config: MiniInferConfig) -> tuple[float, float]:
    if name in {"input_norm", "query", "key", "value"}:
        return 2.0e-4, 2.0e-4
    if name in {"query_rope", "key_rope"}:
        return 3.0e-4, 2.0e-4
    if name in {
        "attention_scores",
        "masked_scores",
        "attention_probabilities",
        "attention_context",
    }:
        return 5.0e-4, 5.0e-4
    accumulation_scale = max(
        1.0, math.ceil(math.log2(max(config.hidden_size, config.intermediate_size)))
    )
    if name in {"output", "mlp_output", "mlp_swiglu"}:
        return 5.0e-4 * accumulation_scale, 1.0e-3
    return 2.5e-4 * accumulation_scale, 5.0e-4


def validate_tensor(
    expected: np.ndarray,
    actual: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    expected_flat = expected.astype(np.float64, copy=False).reshape(-1)
    actual_flat = actual.astype(np.float64, copy=False).reshape(-1)
    if expected_flat.shape != actual_flat.shape:
        raise ValueError("validation shapes differ")

    matching_infinity = (
        np.isinf(expected_flat)
        & np.isinf(actual_flat)
        & (np.signbit(expected_flat) == np.signbit(actual_flat))
    )
    finite = np.isfinite(expected_flat) & np.isfinite(actual_flat)
    absolute_error = np.zeros_like(expected_flat)
    absolute_error[finite] = np.abs(actual_flat[finite] - expected_flat[finite])
    absolute_error[~finite & ~matching_infinity] = np.inf
    threshold = absolute_tolerance + relative_tolerance * np.abs(expected_flat)
    passed_mask = matching_infinity | (finite & (absolute_error <= threshold))
    finite_error = absolute_error[np.isfinite(absolute_error)]
    max_absolute_error = float(np.max(finite_error, initial=0.0))
    mean_absolute_error = float(np.mean(finite_error)) if finite_error.size else 0.0
    worst_index = int(np.argmax(absolute_error)) if absolute_error.size else 0
    meaningful = finite & (np.abs(expected_flat) > absolute_tolerance)
    relative_error = np.zeros_like(expected_flat)
    relative_error[meaningful] = (
        absolute_error[meaningful] / np.abs(expected_flat[meaningful])
    )
    return {
        "passed": bool(np.all(passed_mask)),
        "element_count": int(expected_flat.size),
        "failure_count": int(np.count_nonzero(~passed_mask)),
        "worst_index": worst_index,
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "max_relative_error": float(np.max(relative_error, initial=0.0)),
        "expected_at_worst": float(expected_flat[worst_index]) if expected_flat.size else 0.0,
        "actual_at_worst": float(actual_flat[worst_index]) if actual_flat.size else 0.0,
    }


def validate_intermediates(
    implementation: str,
    fixture_tensors: dict[str, np.ndarray],
    intermediates: dict[str, torch.Tensor],
    config: MiniInferConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    aggregate = {
        "passed": True,
        "element_count": 0,
        "failure_count": 0,
        "worst_index": 0,
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "expected_at_worst": 0.0,
        "actual_at_worst": 0.0,
    }
    absolute_error_sum = 0.0
    index_offset = 0
    for name in INTERMEDIATE_NAMES:
        actual = intermediates[name].detach().cpu().numpy()
        atol, rtol = tolerance(name, config)
        validation = validate_tensor(fixture_tensors[name], actual, atol, rtol)
        records.append(
            {
                "implementation": implementation,
                "name": name,
                "absolute_tolerance": atol,
                "relative_tolerance": rtol,
                **validation,
            }
        )
        aggregate["passed"] = aggregate["passed"] and validation["passed"]
        aggregate["failure_count"] += validation["failure_count"]
        absolute_error_sum += (
            validation["mean_absolute_error"] * validation["element_count"]
        )
        if validation["max_absolute_error"] > aggregate["max_absolute_error"]:
            aggregate["max_absolute_error"] = validation["max_absolute_error"]
            aggregate["worst_index"] = index_offset + validation["worst_index"]
            aggregate["expected_at_worst"] = validation["expected_at_worst"]
            aggregate["actual_at_worst"] = validation["actual_at_worst"]
        aggregate["max_relative_error"] = max(
            aggregate["max_relative_error"], validation["max_relative_error"]
        )
        aggregate["element_count"] += validation["element_count"]
        index_offset += validation["element_count"]
    aggregate["mean_absolute_error"] = (
        absolute_error_sum / aggregate["element_count"]
        if aggregate["element_count"]
        else 0.0
    )
    return records, aggregate


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


def measure_cuda_events(
    model: MiniInferBlock,
    input_device: torch.Tensor,
    warmups: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmups):
        model(input_device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iterations):
        start.record()
        model(input_device)
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples


def measure_end_to_end(
    model: MiniInferBlock,
    input_host: torch.Tensor,
    input_device: torch.Tensor,
    output_host: torch.Tensor,
    warmups: int,
    iterations: int,
) -> list[float]:
    def run_once() -> None:
        input_device.copy_(input_host)
        output = model(input_device)
        output_host.copy_(output)
        torch.cuda.synchronize()

    for _ in range(warmups):
        run_once()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        run_once()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


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


def make_result(
    fixture,
    timing_scope: str,
    samples: list[float],
    validation: dict[str, object],
    warmups: int,
    iterations: int,
    metrics: dict[str, float],
) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    config = fixture.config
    sample_statistics = summarize(samples)
    metrics = {
        **metrics,
        "tokens_per_second_from_median": (
            config.batch
            * config.sequence
            * 1000.0
            / sample_statistics["median"]
        ),
        "validated_intermediate_count": float(len(INTERMEDIATE_NAMES)),
    }
    return {
        "schema_version": 1,
        "config": {
            "warmup_iterations": warmups,
            "measurement_iterations": iterations,
            "seed": fixture.manifest["seed"],
        },
        "metadata": {
            "operation": "miniinfer_block",
            "implementation": "pytorch_cuda_eager",
            "data_type": "fp32",
            "timing_scope": timing_scope,
            "dimensions": {
                "batch": config.batch,
                "sequence": config.sequence,
                "hidden_size": config.hidden_size,
                "attention_heads": config.attention_heads,
                "head_dimension": config.head_dimension,
                "intermediate_size": config.intermediate_size,
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
            "build_type": "Python inference_mode",
            "compiler": f"PyTorch {torch.__version__} / Python {platform.python_version()}",
            "git_commit": git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "statistics_ms": sample_statistics,
        "correctness": validation,
        "metrics": metrics,
        "samples_ms": samples,
    }


def write_outputs(
    output_directory: Path,
    results: list[tuple[str, dict[str, object]]],
    validations: list[dict[str, object]],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, result in results:
        (output_directory / filename).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    with (output_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "schema_version",
                "file",
                "backend",
                "timing_scope",
                "median_ms",
                "p95_ms",
                "tokens_per_second",
                "max_absolute_error",
                "validation_passed",
                "git_commit",
            )
        )
        for filename, result in results:
            writer.writerow(
                (
                    result["schema_version"],
                    filename.removesuffix(".json"),
                    result["metadata"]["implementation"],
                    result["metadata"]["timing_scope"],
                    result["statistics_ms"]["median"],
                    result["statistics_ms"]["p95"],
                    result["metrics"]["tokens_per_second_from_median"],
                    result["correctness"]["max_absolute_error"],
                    str(result["correctness"]["passed"]).lower(),
                    result["metadata"]["git_commit"],
                )
            )
    with (output_directory / "validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        fieldnames = (
            "implementation",
            "name",
            "element_count",
            "absolute_tolerance",
            "relative_tolerance",
            "max_absolute_error",
            "mean_absolute_error",
            "max_relative_error",
            "failure_count",
            "passed",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(validations)


def main() -> None:
    arguments = parse_arguments()
    if arguments.warmups < 0 or arguments.iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")

    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    fixture = load_fixture(arguments.fixture_dir)
    if fixture.manifest["seed"] != 2027:
        raise ValueError("MiniInfer fixture seed must be 2027")
    input_host = torch.from_numpy(fixture.tensors["input"]).clone()

    with torch.inference_mode():
        cpu_model = MiniInferBlock(fixture).eval()
        cpu_intermediates = cpu_model.forward_with_intermediates(input_host)
        cpu_records, cpu_validation = validate_intermediates(
            "pytorch_cpu",
            fixture.tensors,
            cpu_intermediates,
            fixture.config,
        )
        if not cpu_validation["passed"]:
            raise RuntimeError("PyTorch CPU reference reconstruction failed validation")

        model = cpu_model.to("cuda")
        input_device = torch.empty_like(input_host, device="cuda")
        input_device.copy_(input_host)
        gpu_intermediates = model.forward_with_intermediates(input_device)
        gpu_records, gpu_validation = validate_intermediates(
            "pytorch_cuda_eager",
            fixture.tensors,
            gpu_intermediates,
            fixture.config,
        )
        if not gpu_validation["passed"]:
            raise RuntimeError("PyTorch CUDA intermediates failed validation")

        output_host = torch.empty_like(input_host)
        for _ in range(3):
            model(input_device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
        peak_output = model(input_device)
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        del peak_output
        parameter_bytes = sum(
            getattr(model, name).numel() * getattr(model, name).element_size()
            for name in WEIGHT_NAMES
        )
        model_buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in model.buffers()
        )
        common_metrics = {
            "parameter_bytes": float(parameter_bytes),
            "model_buffer_bytes": float(model_buffer_bytes),
            "memory_allocated_before_forward_bytes": float(memory_before),
            "peak_memory_allocated_bytes": float(peak_memory),
            "incremental_peak_forward_bytes": float(max(0, peak_memory - memory_before)),
        }

        kernel_samples = measure_cuda_events(
            model,
            input_device,
            arguments.warmups,
            arguments.iterations,
        )
        end_to_end_samples = measure_end_to_end(
            model,
            input_host,
            input_device,
            output_host,
            arguments.warmups,
            arguments.iterations,
        )

    results = [
        (
            "miniinfer_pytorch_cuda_kernel.json",
            make_result(
                fixture,
                "kernel-sequence-device-resident",
                kernel_samples,
                gpu_validation,
                arguments.warmups,
                arguments.iterations,
                common_metrics,
            ),
        ),
        (
            "miniinfer_pytorch_cuda_end_to_end.json",
            make_result(
                fixture,
                "end-to-end-h2d-forward-d2h",
                end_to_end_samples,
                gpu_validation,
                arguments.warmups,
                arguments.iterations,
                common_metrics,
            ),
        ),
    ]
    write_outputs(arguments.output_dir, results, cpu_records + gpu_records)
    for filename, result in results:
        print(
            f"{filename}: median {result['statistics_ms']['median']:.6f} ms, "
            f"p95 {result['statistics_ms']['p95']:.6f} ms"
        )
    print(
        "PyTorch CPU/CUDA intermediate validation: PASS; "
        f"CUDA max error {gpu_validation['max_absolute_error']:.9g}"
    )


if __name__ == "__main__":
    main()
