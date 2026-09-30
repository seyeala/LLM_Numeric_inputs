"""Dataset loading helpers for the CP9 minimum-number task."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch

from llm_numeric_inputs.data.schemas import TaskName
from llm_numeric_inputs.data.validation import validate_manifest


def load_manifest(manifest_path: str | Path) -> dict:
    """Load and validate a CP5 minimum-task manifest."""
    manifest = validate_manifest(manifest_path)
    if TaskName(manifest["task"]) != TaskName.MINIMUM:
        raise ValueError(f"CP9 minimum task requires minimum data, got {manifest['task']!r}.")
    return manifest


def split_path(manifest_path: str | Path, manifest: dict, split: str) -> Path:
    root = Path(manifest_path).parent
    filename = f"{split}.jsonl"
    if filename not in manifest["files"]:
        raise ValueError(f"Split {split!r} is not present in manifest files.")
    return root / filename


def read_minimum_split(manifest_path: str | Path, split: str) -> tuple[list[str], torch.Tensor, dict]:
    """Read one minimum split as text prompts and numeric targets shaped [N, 1]."""
    manifest = load_manifest(manifest_path)
    path = split_path(manifest_path, manifest, split)
    texts: list[str] = []
    targets: list[float] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            texts.append(record["input"])
            targets.append(float(record["output"]))
    return texts, torch.tensor(targets, dtype=torch.float32).view(-1, 1), manifest


def iter_minimum_batches(
    texts: list[str], targets: torch.Tensor, batch_size: int
) -> Iterator[tuple[list[str], torch.Tensor]]:
    """Yield deterministic ordered mini-batches for minimum-task prompts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        yield texts[start:end], targets[start:end]
