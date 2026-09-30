"""Compatibility wrapper for the pre-CP5 dataset generator.

The original file was exported from a notebook and generated large JSONL files at import time.
CP5 removed those side effects. New code should use scripts/generate_dataset.py or
llm_numeric_inputs.data.generate_dataset directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_numeric_inputs.data.generators import generate_records
from llm_numeric_inputs.data.schemas import TaskName


def _write_single_jsonl(task: TaskName, count: int, filename: str, seed: int) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path("") else None
    with path.open("w", encoding="utf-8") as file:
        for record in generate_records(task, count=count, seed=seed):
            file.write(json.dumps(record, sort_keys=True) + "\n")


def generate_and_save_jsonl_alignment_numbertonumber(n: int, filename: str, seed: int = 42) -> None:
    _write_single_jsonl(TaskName.NUMERIC_TO_NUMERIC, n, filename, seed)


def generate_and_save_jsonl_alignment_textnumber(n: int, filename: str, seed: int = 42) -> None:
    _write_single_jsonl(TaskName.TEXT_TO_NUMERIC, n, filename, seed)


def generate_questions_and_answers(m: int, filename: str, seed: int = 42) -> None:
    _write_single_jsonl(TaskName.MINIMUM, m, filename, seed)


if __name__ == "__main__":
    raise SystemExit(
        "This compatibility wrapper no longer generates files at import time. "
        "Use scripts/generate_dataset.py for deterministic CP5 datasets."
    )
