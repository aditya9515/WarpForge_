#!/usr/bin/env python3
"""Validate Stage 10 benchmark records and write their comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_FILES = (
    "native/miniinfer_custom_kernel.json",
    "native/miniinfer_custom_end_to_end.json",
    "native/miniinfer_cublas_kernel.json",
    "native/miniinfer_cublas_end_to_end.json",
    "pytorch/miniinfer_pytorch_cuda_kernel.json",
    "pytorch/miniinfer_pytorch_cuda_end_to_end.json",
    "tensorrt/miniinfer_tensorrt_fp32_kernel.json",
    "tensorrt/miniinfer_tensorrt_fp32_transfer.json",
    "tensorrt/miniinfer_tensorrt_fp16_kernel.json",
    "tensorrt/miniinfer_tensorrt_fp16_transfer.json",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    rows: list[dict[str, object]] = []
    commits: set[str] = set()
    for relative_path in EXPECTED_FILES:
        path = arguments.result_directory / relative_path
        result = json.loads(path.read_text(encoding="utf-8"))
        metadata = result["metadata"]
        statistics = result["statistics_ms"]
        correctness = result["correctness"]
        dimensions = metadata["dimensions"]
        if result["schema_version"] != 1:
            raise ValueError(f"{relative_path}: unsupported schema")
        if dimensions != {
            "batch": 1,
            "sequence": 128,
            "hidden_size": 512,
            "attention_heads": 8,
            "head_dimension": 64,
            "intermediate_size": 1536,
        }:
            raise ValueError(f"{relative_path}: unexpected dimensions")
        if statistics["sample_count"] != 500 or len(result["samples_ms"]) != 500:
            raise ValueError(f"{relative_path}: expected 500 samples")
        if result["config"]["seed"] != 2027 or not correctness["passed"]:
            raise ValueError(f"{relative_path}: reproducibility/correctness check failed")
        commits.add(metadata["git_commit"])
        rows.append(
            {
                "schema_version": 1,
                "file": relative_path.removesuffix(".json"),
                "backend": metadata["implementation"],
                "data_type": metadata["data_type"],
                "timing_scope": metadata["timing_scope"],
                "median_ms": statistics["median"],
                "p95_ms": statistics["p95"],
                "tokens_per_second": result["metrics"][
                    "tokens_per_second_from_median"
                ],
                "max_absolute_error": correctness["max_absolute_error"],
                "validation_passed": "true",
                "git_commit": metadata["git_commit"],
            }
        )
    if len(commits) != 1:
        raise ValueError(f"results do not share one clean code commit: {sorted(commits)}")

    output_path = arguments.result_directory / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Stage 10 results: PASS ({len(rows)} records, commit {commits.pop()})")


if __name__ == "__main__":
    main()
