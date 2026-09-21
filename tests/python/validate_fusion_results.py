import csv
import json
import pathlib
import sys


EXPECTED_FILES = {
    "pipeline_single_stream.json",
    "pipeline_two_stream.json",
    "residual_rmsnorm_fused.json",
    "residual_rmsnorm_separate.json",
    "swiglu_fused.json",
    "swiglu_separate.json",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_fusion_results.py <result-directory>")

    directory = pathlib.Path(sys.argv[1])
    paths = sorted(directory.glob("*.json"))
    assert {path.name for path in paths} == EXPECTED_FILES
    for path in paths:
        with path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
        assert result["schema_version"] == 1
        expected_scope = "end-to-end" if path.name.startswith("pipeline_") else "kernel-only"
        assert result["metadata"]["timing_scope"] == expected_scope
        assert result["metadata"]["data_type"] == "fp32"
        assert result["correctness"]["passed"] is True
        assert result["correctness"]["failure_count"] == 0
        assert result["statistics_ms"]["sample_count"] == result["config"]["measurement_iterations"]
        assert len(result["samples_ms"]) == result["config"]["measurement_iterations"]
        assert result["metrics"]["effective_bandwidth_gbps_from_median"] > 0.0
        assert result["metrics"]["speedup_vs_operation_baseline"] > 0.0
        assert result["metrics"]["logical_reads_per_element"] > 0.0
        assert result["metrics"]["logical_writes_per_element"] > 0.0

    with (directory / "summary.csv").open("r", encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert len(rows) == len(EXPECTED_FILES)
    assert all(row["validation_passed"] == "true" for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
