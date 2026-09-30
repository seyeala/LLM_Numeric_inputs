"""Dataset loading and tokenization helpers for text-to-numeric experiments."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch

from llm_numeric_inputs.data.schemas import TaskName
from llm_numeric_inputs.data.validation import validate_manifest


def load_text_manifest(manifest_path: str | Path) -> dict:
    """Load and validate a CP5 text_to_numeric manifest."""
    manifest = validate_manifest(manifest_path)
    if TaskName(manifest["task"]) != TaskName.TEXT_TO_NUMERIC:
        raise ValueError(f"CP7 text alignment requires text_to_numeric data, got {manifest['task']!r}.")
    return manifest


def text_split_path(manifest_path: str | Path, manifest: dict, split: str) -> Path:
    root = Path(manifest_path).parent
    filename = f"{split}.jsonl"
    if filename not in manifest["files"]:
        raise ValueError(f"Split {split!r} is not present in manifest files.")
    return root / filename


def read_text_split(manifest_path: str | Path, split: str) -> tuple[list[str], torch.Tensor, dict]:
    """Read one text_to_numeric split into text strings and target tensor [N, 1]."""
    manifest = load_text_manifest(manifest_path)
    path = text_split_path(manifest_path, manifest, split)
    texts: list[str] = []
    targets: list[float] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            texts.append(str(record["input"]))
            targets.append(float(record["output"]))
    y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    return texts, y, manifest


def iter_text_batches(
    texts: list[str], targets: torch.Tensor, batch_size: int
) -> Iterator[tuple[list[str], torch.Tensor]]:
    """Yield deterministic ordered text mini-batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        yield texts[start:end], targets[start:end]


def tokenize_text_batch(tokenizer, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    """Tokenize a text batch and move tensor outputs to the requested device."""
    tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    if hasattr(tokenized, "to"):
        return tokenized.to(device)
    return {key: value.to(device) for key, value in tokenized.items() if torch.is_tensor(value)}
