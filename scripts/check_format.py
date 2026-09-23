"""Check every committed C++/CUDA file against the pinned .clang-format style."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".cpp", ".cu", ".cuh", ".hpp", ".h"}


def main() -> int:
    formatter = shutil.which("clang-format")
    if formatter is None:
        print("clang-format is required for the formatting check", file=sys.stderr)
        return 2
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    files = [
        name.decode("utf-8")
        for name in output.split(b"\0")
        if name and Path(name.decode("utf-8")).suffix.lower() in SOURCE_SUFFIXES
    ]
    result = subprocess.run(
        [formatter, "--dry-run", "--Werror", "--style=file", *files],
        cwd=ROOT,
        check=False,
    )
    if result.returncode == 0:
        print(f"clang-format check passed for {len(files)} C++/CUDA files")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
