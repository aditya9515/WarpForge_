"""Make a small release plot from committed, correctness-gated Stage 11 CSVs.

This is a derived visualization, not a new benchmark run. It uses only the Python
standard library so CPU-only CI can verify the inputs without GPU access.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "stage11"


def selected_results() -> list[dict[str, str | float]]:
    cases = [
        ("FP32 sum", "reduction", "variant", "naive_interleaved", "warp_shuffle", "reduction_sum"),
        ("FP32 GEMM", "gemm", "implementation", "custom_naive_fp32", "custom_register_blocked_fp32", None),
        ("Softmax", "transformer", "implementation", "naive", "warp", "softmax"),
        ("RMSNorm", "transformer", "implementation", "naive", "block", "rmsnorm"),
    ]
    selected = []
    for label, directory, field, baseline, optimized, operation in cases:
        source = RESULTS / directory / "summary.csv"
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidates = [row for row in rows if operation is None or row.get("operation") == operation]
        baseline_rows = [row for row in candidates if row[field] == baseline]
        optimized_rows = [row for row in candidates if row[field] == optimized]
        if len(baseline_rows) != 1 or len(optimized_rows) != 1:
            raise ValueError(f"missing or ambiguous {label} rows in {source}")
        first, second = baseline_rows[0], optimized_rows[0]
        if first["validation_passed"] != "true" or second["validation_passed"] != "true":
            raise ValueError(f"unvalidated {label} record")
        if first["git_commit"] != second["git_commit"]:
            raise ValueError(f"mixed source commits for {label}")
        baseline_ms = float(first["median_ms"])
        optimized_ms = float(second["median_ms"])
        if baseline_ms <= 0 or optimized_ms <= 0:
            raise ValueError(f"non-positive latency for {label}")
        selected.append(
            {
                "workload": label,
                "baseline": baseline,
                "optimized": optimized,
                "baseline_median_ms": baseline_ms,
                "optimized_median_ms": optimized_ms,
                "speedup": baseline_ms / optimized_ms,
                "source_commit": first["git_commit"],
                "source_csv": str(source.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    if len({row["source_commit"] for row in selected}) != 1:
        raise ValueError("selected workloads do not share one clean source commit")
    return selected


def render_summary(rows: list[dict[str, str | float]]) -> str:
    fields = [
        "workload", "baseline", "optimized", "baseline_median_ms",
        "optimized_median_ms", "speedup", "source_commit", "source_csv",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: f"{row[key]:.6f}" if isinstance(row[key], float) else row[key] for key in fields}
        )
    return output.getvalue()


def render_svg(rows: list[dict[str, str | float]]) -> str:
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="365" viewBox="0 0 840 365" role="img" aria-labelledby="title description">',
        '<title id="title">WarpForge baseline-to-optimized median speedups</title>',
        '<desc id="description">Four CUDA-event benchmark comparisons on the RTX 3050 Laptop GPU.</desc>',
        '<rect width="840" height="365" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172536}.heading{font-size:23px;font-weight:700}.sub{font-size:13px;fill:#536477}.label{font-size:16px;font-weight:700}.value{font-size:15px;font-weight:700}.foot{font-size:12px;fill:#536477}</style>',
        '<text x="35" y="40" class="heading">Optimization ladders: measured median speedup</text>',
        '<text x="35" y="64" class="sub">FP32 · RTX 3050 Laptop GPU · 50 warmups / 500 samples · kernel-only timing</text>',
    ]
    for index, row in enumerate(rows):
        y = 108 + index * 56
        width = float(row["speedup"]) * 80.0
        elements.extend(
            [
                f'<text x="35" y="{y + 17}" class="label">{escape(str(row["workload"]))}</text>',
                f'<rect x="180" y="{y}" width="{width:.2f}" height="25" rx="4" fill="#087e79"/>',
                f'<text x="{190 + width:.2f}" y="{y + 18}" class="value">{float(row["speedup"]):.2f}×</text>',
            ]
        )
    elements.extend(
        [
            '<text x="35" y="337" class="foot">Derived from committed Stage 11 CSVs; this is not a new Stage 12 timing run.</text>',
            '</svg>',
        ]
    )
    return "\n".join(elements) + "\n"


def write_summary(rows: list[dict[str, str | float]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write(render_summary(rows))


def write_svg(rows: list[dict[str, str | float]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_svg(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "stage12" / "release_summary.csv",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "docs" / "performance" / "stage12_latency.svg",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed CSV/SVG exactly match the deterministic generator",
    )
    arguments = parser.parse_args()
    rows = selected_results()
    if arguments.check:
        if render_summary(rows) != arguments.summary.read_text(encoding="utf-8"):
            raise ValueError(f"stale release summary: {arguments.summary}")
        if render_svg(rows) != arguments.plot.read_text(encoding="utf-8"):
            raise ValueError(f"stale release plot: {arguments.plot}")
        print("Committed release summary and plot match Stage 11 measurements")
        return
    write_summary(rows, arguments.summary)
    write_svg(rows, arguments.plot)
    print(f"Wrote {len(rows)} validated comparisons to {arguments.summary} and {arguments.plot}")


if __name__ == "__main__":
    main()
