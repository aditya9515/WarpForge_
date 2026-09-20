import csv
import json
import pathlib
import sys


OPERATIONS = ("sum", "maximum")
VARIANTS = (
    "naive_interleaved",
    "shared_memory",
    "reduced_divergence",
    "unrolled",
    "warp_shuffle",
)
EXPECTED_FILES = {
    f"{operation}_{variant}.json"
    for operation in OPERATIONS
    for variant in VARIANTS
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_reduction_results.py <result-directory>")

    directory = pathlib.Path(sys.argv[1])
    actual_files = {path.name for path in directory.glob("*.json")}
    assert actual_files == EXPECTED_FILES, (actual_files, EXPECTED_FILES)

    for path in directory.glob("*.json"):
        with path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
        assert result["schema_version"] == 1
        assert result["metadata"]["operation"] in {
            "reduction_sum",
            "reduction_maximum",
        }
        assert result["metadata"]["implementation"] in VARIANTS
        assert result["metadata"]["timing_scope"] == "kernel-only"
        assert result["correctness"]["passed"] is True
        assert result["correctness"]["failure_count"] == 0
        assert result["statistics_ms"]["sample_count"] == result["config"]["measurement_iterations"]
        assert len(result["samples_ms"]) == result["config"]["measurement_iterations"]
        assert result["metrics"]["effective_input_bandwidth_gbps_from_median"] >= 0.0
        assert result["metrics"]["speedup_vs_naive_interleaved"] > 0.0

    with (directory / "summary.csv").open("r", encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert len(rows) == len(EXPECTED_FILES)
    assert all(row["validation_passed"] == "true" for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
