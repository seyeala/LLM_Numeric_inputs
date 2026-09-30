"""Evaluation helpers for the CP9 minimum-number task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from llm_numeric_inputs.experiments.metrics import regression_metrics
from llm_numeric_inputs.experiments.minimum_data import read_minimum_split


def predict_minimum(
    model: torch.nn.Module, texts: list[str], batch_size: int, device: torch.device
) -> torch.Tensor:
    """Run deterministic batched prediction for mixed minimum-task prompts."""
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            outputs = model({"input_text": batch}).detach().cpu()
            predictions.append(outputs)
    return torch.cat(predictions, dim=0) if predictions else torch.empty(0, 1)


def evaluate_minimum_alignment(
    model: torch.nn.Module,
    manifest_path: str | Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a mixed minimum-task model on one CP5 split."""
    texts, targets, manifest = read_minimum_split(manifest_path, split)
    predictions = predict_minimum(model, texts, batch_size, device)
    lower, upper = manifest.get("numeric_range", [0.0, 100.0])
    metrics = regression_metrics(predictions, targets, (float(lower), float(upper)))
    metrics.update({"split": split, "task": manifest["task"], "dataset_count": len(texts)})
    return metrics
