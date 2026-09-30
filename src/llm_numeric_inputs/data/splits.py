"""Deterministic split-count utilities."""

from __future__ import annotations

from llm_numeric_inputs.data.schemas import SPLIT_NAMES


def split_counts(
    count: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, int]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0.")
    train_count = int(count * train_ratio)
    validation_count = int(count * validation_ratio)
    test_count = count - train_count - validation_count
    return {"train": train_count, "validation": validation_count, "test": test_count}


def split_for_index(index: int, counts: dict[str, int]) -> str:
    if index < 0:
        raise ValueError("index must be non-negative.")
    offset = 0
    for split_name in SPLIT_NAMES:
        offset += counts[split_name]
        if index < offset:
            return split_name
    raise IndexError("index is outside the split counts.")
