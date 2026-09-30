"""Regression metrics for numeric-input experiments."""

from __future__ import annotations

from typing import Any

import torch


def _as_flat_float_tensor(values: torch.Tensor | list[float]) -> torch.Tensor:
    tensor = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
    return tensor.detach().to(torch.float32).reshape(-1).cpu()


def regression_metrics(
    predictions: torch.Tensor | list[float],
    targets: torch.Tensor | list[float],
    value_range: tuple[float, float] = (0.0, 100.0),
) -> dict[str, Any]:
    """Return deterministic scalar regression metrics for predictions and targets."""
    pred = _as_flat_float_tensor(predictions)
    tgt = _as_flat_float_tensor(targets)
    if pred.numel() != tgt.numel():
        raise ValueError(f"predictions and targets must have the same count: {pred.numel()} != {tgt.numel()}")
    if pred.numel() == 0:
        raise ValueError("at least one prediction/target pair is required.")
    errors = pred - tgt
    abs_errors = errors.abs()
    mse = torch.mean(errors.square()).item()
    mae = torch.mean(abs_errors).item()
    rmse = mse**0.5
    lower, upper = value_range
    span = float(upper) - float(lower)
    if span <= 0:
        raise ValueError("value_range must have positive width.")
    nonzero = tgt.abs().clamp_min(1e-12)
    return {
        "count": int(pred.numel()),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mean_absolute_relative_error": float(torch.mean(abs_errors / nonzero).item()),
        "normalized_mae_over_range": float(mae / span),
        "p50_abs_error": float(torch.quantile(abs_errors, 0.50).item()),
        "p90_abs_error": float(torch.quantile(abs_errors, 0.90).item()),
        "p95_abs_error": float(torch.quantile(abs_errors, 0.95).item()),
        "max_abs_error": float(torch.max(abs_errors).item()),
    }
