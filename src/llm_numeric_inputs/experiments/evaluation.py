"""Evaluation helpers for numeric alignment experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from llm_numeric_inputs.experiments.metrics import regression_metrics
from llm_numeric_inputs.experiments.numeric_data import read_numeric_split


def predict_numeric(
    model: torch.nn.Module, inputs: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Run deterministic batched prediction for numeric tensors."""
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size].to(device)
            predictions.append(model(batch).detach().cpu())
    return torch.cat(predictions, dim=0) if predictions else torch.empty(0, 1)


def evaluate_numeric_alignment(
    model: torch.nn.Module,
    manifest_path: str | Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a numeric alignment model on one CP5 split."""
    inputs, targets, manifest = read_numeric_split(manifest_path, split)
    predictions = predict_numeric(model, inputs, batch_size, device)
    lower, upper = manifest.get("numeric_range", [0.0, 100.0])
    metrics = regression_metrics(predictions, targets, (float(lower), float(upper)))
    metrics.update({"split": split, "task": manifest["task"], "dataset_count": int(inputs.shape[0])})
    return metrics
