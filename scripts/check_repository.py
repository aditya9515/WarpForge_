"""Dependency-free checks for committed docs, JSON, and generated-artifact policy."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
RAW_SUFFIXES = {".ncu-rep", ".nsys-rep", ".sqlite", ".engine", ".onnx"}
GENERATED_DIRS = {"build", "out", ".venv", ".venv-stage10", ".stage10-tools"}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if target.startswith("#"):
        return None
    split = urlsplit(target)
    if split.scheme or split.netloc:
        return None
    path = unquote(split.path)
    if not path:
        return None
    return (source.parent / path).resolve()


def main() -> int:
    problems: list[str] = []
    files = tracked_files()
    markdown_count = 0
    json_count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts[0] in GENERATED_DIRS or path.suffix.lower() in RAW_SUFFIXES:
            problems.append(f"generated artifact is tracked: {relative}")
        if path.suffix.lower() == ".json":
            json_count += 1
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                problems.append(f"invalid JSON {relative}: {error}")
        if path.suffix.lower() == ".md":
            markdown_count += 1
            try:
                contents = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                problems.append(f"unreadable Markdown {relative}: {error}")
                continue
            for match in LINK.finditer(contents):
                destination = local_link_target(path, match.group(1))
                if destination is not None and not destination.is_file():
                    problems.append(f"broken link {relative}: {match.group(1)}")

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"Repository checks passed: {markdown_count} Markdown files, {json_count} JSON files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
