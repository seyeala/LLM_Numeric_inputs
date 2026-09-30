"""Dataset loading helpers for numeric alignment experiments."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch

from llm_numeric_inputs.data.schemas import TaskName
from llm_numeric_inputs.data.validation import validate_manifest


def load_manifest(manifest_path: str | Path) -> dict:
    """Load and validate a CP5 manifest."""
    manifest = validate_manifest(manifest_path)
    if TaskName(manifest["task"]) != TaskName.NUMERIC_TO_NUMERIC:
        raise ValueError(f"CP6 numeric alignment requires numeric_to_numeric data, got {manifest['task']!r}.")
    return manifest


def split_path(manifest_path: str | Path, manifest: dict, split: str) -> Path:
    root = Path(manifest_path).parent
    filename = f"{split}.jsonl"
    if filename not in manifest["files"]:
        raise ValueError(f"Split {split!r} is not present in manifest files.")
    return root / filename


def read_numeric_split(manifest_path: str | Path, split: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Read one numeric_to_numeric split into tensors shaped [N, 1]."""
    manifest = load_manifest(manifest_path)
    path = split_path(manifest_path, manifest, split)
    inputs: list[float] = []
    targets: list[float] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            inputs.append(float(record["input"]))
            targets.append(float(record["output"]))
    x = torch.tensor(inputs, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    return x, y, manifest


def iter_batches(
    inputs: torch.Tensor, targets: torch.Tensor, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield deterministic ordered mini-batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        yield inputs[start:end], targets[start:end]
