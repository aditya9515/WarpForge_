import csv
import json
import pathlib
import sys


IMPLEMENTATIONS = {
    "cublas_fp32",
    "custom_naive_fp32",
    "custom_tiled_fp32",
    "custom_coalesced_fp32",
    "custom_register_blocked_fp32",
    "cublas_fp16_fp32",
    "custom_tiled_fp16_fp32",
    "custom_wmma_fp16_fp32",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_gemm_results.py <result-directory>")

    directory = pathlib.Path(sys.argv[1])
    paths = sorted(directory.glob("*.json"))
    assert paths, "no GEMM JSON files found"
    seen_by_size: dict[int, set[str]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
        assert result["schema_version"] == 1
        metadata = result["metadata"]
        assert metadata["operation"] in {"gemm_fp32", "gemm_fp16_fp32"}
        assert metadata["implementation"] in IMPLEMENTATIONS
        assert metadata["timing_scope"] == "kernel-only"
        dimensions = metadata["dimensions"]
        assert dimensions["m"] > 0
        assert dimensions["n"] > 0
        assert dimensions["k"] > 0
        correctness = result["correctness"]
        assert correctness["passed"] is True
        assert correctness["failure_count"] == 0
        assert result["statistics_ms"]["sample_count"] == result["config"]["measurement_iterations"]
        assert len(result["samples_ms"]) == result["config"]["measurement_iterations"]
        assert result["metrics"]["gflops_from_median"] > 0.0
        assert result["metrics"]["percent_of_cublas"] > 0.0
        seen_by_size.setdefault(dimensions["m"], set()).add(metadata["implementation"])

    assert all(implementations == IMPLEMENTATIONS for implementations in seen_by_size.values())
    with (directory / "summary.csv").open("r", encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert len(rows) == len(paths)
    assert all(row["validation_passed"] == "true" for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
