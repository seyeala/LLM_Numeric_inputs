"""High-level deterministic dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_numeric_inputs.data.generators import generate_records
from llm_numeric_inputs.data.manifests import build_manifest, write_manifest
from llm_numeric_inputs.data.schemas import SPLIT_NAMES, TaskName
from llm_numeric_inputs.data.splits import split_counts, split_for_index
from llm_numeric_inputs.data.validation import validate_manifest


def generate_dataset(
    task: TaskName | str,
    count: int,
    seed: int,
    output_dir: str | Path,
    min_val: float = 0.0,
    max_val: float = 100.0,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    numeric_decimals: int = 4,
    minimum_decimals: int = 2,
    min_terms: int = 1,
    max_terms: int = 10,
    validate: bool = True,
) -> Path:
    """Generate split JSONL files and a manifest for one deterministic dataset."""
    task_name = TaskName(task)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    counts = split_counts(count, train_ratio, validation_ratio, test_ratio)
    split_paths = {split: output_path / f"{split}.jsonl" for split in SPLIT_NAMES}
    handles = {split: path.open("w", encoding="utf-8") for split, path in split_paths.items()}
    try:
        for index, record in enumerate(
            generate_records(
                task_name,
                count,
                seed,
                min_val=min_val,
                max_val=max_val,
                numeric_decimals=numeric_decimals,
                minimum_decimals=minimum_decimals,
                min_terms=min_terms,
                max_terms=max_terms,
            )
        ):
            split_name = split_for_index(index, counts)
            handles[split_name].write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    manifest = build_manifest(
        output_path,
        task_name,
        seed,
        count,
        counts,
        min_val,
        max_val,
        split_paths,
        extra={
            "numeric_decimals": numeric_decimals,
            "minimum_decimals": minimum_decimals,
            "min_terms": min_terms,
            "max_terms": max_terms,
            "split_ratios": {
                "train": train_ratio,
                "validation": validation_ratio,
                "test": test_ratio,
            },
        },
    )
    manifest_path = write_manifest(manifest, output_path / "manifest.json")
    if validate:
        validate_manifest(manifest_path)
    return manifest_path


def manifest_summary(manifest: dict[str, Any]) -> str:
    return f"task={manifest['task']} count={manifest['count']} seed={manifest['seed']}"
