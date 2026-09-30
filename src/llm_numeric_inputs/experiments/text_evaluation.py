"""Evaluation helpers for text-to-numeric alignment experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from llm_numeric_inputs.experiments.metrics import regression_metrics
from llm_numeric_inputs.experiments.text_data import iter_text_batches, read_text_split, tokenize_text_batch


def predict_text_numeric(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run deterministic batched prediction for tokenized text inputs."""
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch_texts, _ in iter_text_batches(texts, torch.empty(len(texts), 1), batch_size):
            tokenized = tokenize_text_batch(tokenizer, batch_texts, device)
            predictions.append(model(tokenized).detach().cpu())
    return torch.cat(predictions, dim=0) if predictions else torch.empty(0, 1)


def evaluate_text_alignment(
    model: torch.nn.Module,
    tokenizer,
    manifest_path: str | Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a text-to-numeric model on one CP5 split."""
    texts, targets, manifest = read_text_split(manifest_path, split)
    predictions = predict_text_numeric(model, tokenizer, texts, batch_size, device)
    lower, upper = manifest.get("numeric_range", [0.0, 100.0])
    metrics = regression_metrics(predictions, targets, (float(lower), float(upper)))
    metrics.update({"split": split, "task": manifest["task"], "dataset_count": int(targets.shape[0])})
    return metrics
