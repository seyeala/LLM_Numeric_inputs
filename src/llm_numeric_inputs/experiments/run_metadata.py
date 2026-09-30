"""Run metadata and artifact helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_sha() -> str | None:
    return command_output(["git", "rev-parse", "HEAD"])


def git_status_porcelain() -> str | None:
    return command_output(["git", "status", "--porcelain"])


def new_run_id(prefix: str = "cp6_numeric") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = (git_sha() or "nogit")[:8]
    return f"{prefix}_{stamp}_{sha}"


def environment_metadata() -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "numpy": package_version("numpy"),
            "PyYAML": package_version("PyYAML"),
        },
        "torch_runtime": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "devices": cuda_devices,
        },
        "git_sha": git_sha(),
        "git_status_porcelain": git_status_porcelain(),
    }


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
