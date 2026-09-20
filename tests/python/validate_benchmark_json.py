import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate_benchmark_json.py <result.json> <schema.json>")

    path = pathlib.Path(sys.argv[1])
    schema_path = pathlib.Path(sys.argv[2])
    with path.open("r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    with schema_path.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    assert schema["properties"]["schema_version"]["const"] == 1
    assert result["schema_version"] == 1
    assert result["metadata"]["operation"] == "vector_add"
    assert result["metadata"]["timing_scope"] == "kernel-only"
    assert result["correctness"]["passed"] is True
    assert result["statistics_ms"]["sample_count"] == result["config"]["measurement_iterations"]
    assert len(result["samples_ms"]) == result["config"]["measurement_iterations"]
    assert result["metrics"]["effective_bandwidth_gbps_from_median"] >= 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
