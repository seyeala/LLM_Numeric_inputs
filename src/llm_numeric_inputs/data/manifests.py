"""Dataset manifest and file-hash helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from llm_numeric_inputs.data.schemas import SCHEMA_VERSION, TaskName


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    output_dir: str | Path,
    task: TaskName | str,
    seed: int,
    count: int,
    counts: dict[str, int],
    min_val: float,
    max_val: float,
    files: dict[str, Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    file_info = {}
    for split_name, path in files.items():
        relative = path.relative_to(root).as_posix()
        file_info[relative] = {
            "split": split_name,
            "rows": count_lines(path),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": str(TaskName(task)),
        "seed": seed,
        "count": count,
        "counts": counts,
        "numeric_range": [min_val, max_val],
        "generator": {
            "git_sha": git_sha(),
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
        },
        "files": file_info,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
