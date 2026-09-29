"""Print a machine-readable runtime environment report for experiment provenance."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


report = {
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "packages": {
        name: package_version(name)
        for name in ("torch", "transformers", "numpy", "PyYAML", "pytest", "ruff")
    },
}

try:
    import torch

    report["torch_runtime"] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }
except (ImportError, OSError, RuntimeError) as error:
    report["torch_runtime"] = {"import_error": repr(error)}

report["nvidia_smi"] = command_output(
    [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
)
report["git_sha"] = command_output(["git", "rev-parse", "HEAD"])
report["git_status_porcelain"] = command_output(["git", "status", "--porcelain"])

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="Optional JSON output path.")
args = parser.parse_args()
payload = json.dumps(report, indent=2) + "\n"

if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
else:
    print(payload, end="")
