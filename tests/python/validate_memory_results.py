import csv
import json
import pathlib
import sys


EXPECTED_FILES = {
    *(f"saxpy_block_{block}.json" for block in (32, 64, 128, 256, 512)),
    *(f"copy_block_{block}.json" for block in (32, 64, 128, 256, 512)),
    *(f"strided_stride_{stride}.json" for stride in (1, 2, 4, 8, 16, 32)),
    "transpose_naive.json",
    "transpose_tiled_32x32.json",
    "transfer_pageable_sync.json",
    "transfer_pinned_sync.json",
    "transfer_pageable_async.json",
    "transfer_pinned_async.json",
    "pipeline_single_stream_chunked.json",
    "pipeline_two_stream_chunked.json",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_memory_results.py <result-directory>")

    directory = pathlib.Path(sys.argv[1])
    actual_files = {path.name for path in directory.glob("*.json")}
    assert actual_files == EXPECTED_FILES, (actual_files, EXPECTED_FILES)

    for path in directory.glob("*.json"):
        with path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
        assert result["schema_version"] == 1
        assert result["correctness"]["passed"] is True
        assert result["correctness"]["failure_count"] == 0
        assert result["statistics_ms"]["sample_count"] == result["config"]["measurement_iterations"]
        assert len(result["samples_ms"]) == result["config"]["measurement_iterations"]
        assert result["metadata"]["timing_scope"] in {"kernel-only", "end-to-end"}
        assert result["metrics"]["effective_bandwidth_gbps_from_median"] >= 0.0

    with (directory / "summary.csv").open("r", encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert len(rows) == len(EXPECTED_FILES)
    assert all(row["schema_version"] == "1" for row in rows)
    assert all(row["validation_passed"] == "true" for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
