#!/usr/bin/env python3

import json
import csv
import sys
from pathlib import Path


EXPECTED = {
    "miniinfer_custom_kernel.json": ("custom", "kernel-sequence-device-resident"),
    "miniinfer_custom_end_to_end.json": ("custom", "end-to-end-h2d-forward-d2h"),
    "miniinfer_cublas_kernel.json": ("cublas", "kernel-sequence-device-resident"),
    "miniinfer_cublas_end_to_end.json": ("cublas", "end-to-end-h2d-forward-d2h"),
}


def validate(directory: Path) -> None:
    for filename, (backend, timing_scope) in EXPECTED.items():
        result = json.loads((directory / filename).read_text(encoding="utf-8"))
        assert result["schema_version"] == 1
        assert result["metadata"]["operation"] == "miniinfer_block"
        assert result["metadata"]["implementation"] == backend
        assert result["metadata"]["timing_scope"] == timing_scope
        assert result["metadata"]["data_type"] == "fp32"
        assert result["statistics_ms"]["sample_count"] == result["config"]["measurement_iterations"]
        assert result["statistics_ms"]["median"] > 0.0
        assert result["metrics"]["tokens_per_second_from_median"] > 0.0
        assert result["metrics"]["validated_intermediate_count"] == 18.0
        assert result["correctness"]["passed"] is True
        assert result["correctness"]["failure_count"] == 0
        assert len(result["samples_ms"]) == result["config"]["measurement_iterations"]

    with (directory / "validation.csv").open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 36
    assert {row["backend"] for row in rows} == {"custom", "cublas"}
    assert all(row["passed"] == "true" for row in rows)
    assert all(int(row["failure_count"]) == 0 for row in rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_miniinfer_results.py <result-directory>")
    validate(Path(sys.argv[1]))
