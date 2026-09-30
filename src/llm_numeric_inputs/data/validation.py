"""Streaming JSONL validation for deterministic datasets."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from llm_numeric_inputs.data.manifests import count_lines, sha256_file
from llm_numeric_inputs.data.schemas import REQUIRED_RECORD_KEYS, TaskName

TEXT_NUMERIC_PATTERN = re.compile(r"^\$\$(.*?)&&$")
MARKER_PATTERN = re.compile(r"\$\$(.*?)&&")


class DatasetValidationError(ValueError):
    """Raised when a dataset file or manifest is invalid."""


def _decimal(text: str, context: str) -> Decimal:
    try:
        return Decimal(str(text))
    except (InvalidOperation, ValueError) as error:
        raise DatasetValidationError(f"Invalid decimal for {context}: {text!r}") from error


def _range_check(value: Decimal, min_val: float, max_val: float, context: str) -> None:
    if not (Decimal(str(min_val)) <= value <= Decimal(str(max_val))):
        raise DatasetValidationError(f"{context} value {value} outside [{min_val}, {max_val}].")


def _require_schema(record: Any, line_number: int) -> dict[str, str]:
    if not isinstance(record, dict):
        raise DatasetValidationError(f"line {line_number}: record must be a JSON object.")
    missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        raise DatasetValidationError(f"line {line_number}: missing keys {missing}.")
    if not all(isinstance(record[key], str) for key in REQUIRED_RECORD_KEYS):
        raise DatasetValidationError(f"line {line_number}: input and output must be strings.")
    return record


def validate_record(record: dict[str, str], task: TaskName | str, min_val: float = 0.0, max_val: float = 100.0) -> None:
    task_name = TaskName(task)
    if task_name == TaskName.NUMERIC_TO_NUMERIC:
        input_value = _decimal(record["input"], "numeric input")
        output_value = _decimal(record["output"], "numeric output")
        if input_value != output_value:
            raise DatasetValidationError(f"numeric input/output mismatch: {input_value} != {output_value}")
        _range_check(input_value, min_val, max_val, "numeric input")
        return
    if task_name == TaskName.TEXT_TO_NUMERIC:
        match = TEXT_NUMERIC_PATTERN.match(record["input"])
        if not match:
            raise DatasetValidationError(f"text-to-numeric input has invalid marker syntax: {record['input']!r}")
        input_value = _decimal(match.group(1), "text numeric marker")
        output_value = _decimal(record["output"], "numeric output")
        if input_value != output_value:
            raise DatasetValidationError(f"text numeric input/output mismatch: {input_value} != {output_value}")
        _range_check(input_value, min_val, max_val, "text numeric marker")
        return
    if task_name == TaskName.MINIMUM:
        values = [_decimal(value, "minimum marker") for value in MARKER_PATTERN.findall(record["input"])]
        if not values:
            raise DatasetValidationError("minimum input contains no numeric markers.")
        output_value = _decimal(record["output"], "minimum output")
        if output_value != min(values):
            raise DatasetValidationError(f"minimum output {output_value} does not equal min({values}).")
        _range_check(output_value, min_val, max_val, "minimum output")
        return
    raise DatasetValidationError(f"Unsupported task: {task!r}")


def validate_jsonl_file(path: str | Path, task: TaskName | str, min_val: float = 0.0, max_val: float = 100.0) -> int:
    rows = 0
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            rows += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(f"line {line_number}: invalid JSON: {error}") from error
            validate_record(_require_schema(raw, line_number), task, min_val, max_val)
    return rows


def validate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    task = TaskName(manifest["task"])
    min_val, max_val = manifest.get("numeric_range", [0.0, 100.0])
    total_rows = 0
    for relative_path, info in manifest["files"].items():
        file_path = root / relative_path
        rows = validate_jsonl_file(file_path, task, min_val, max_val)
        if rows != info["rows"]:
            raise DatasetValidationError(f"{relative_path}: row count {rows} != manifest {info['rows']}")
        if count_lines(file_path) != info["rows"]:
            raise DatasetValidationError(f"{relative_path}: line count mismatch.")
        digest = sha256_file(file_path)
        if digest != info["sha256"]:
            raise DatasetValidationError(f"{relative_path}: sha256 {digest} != manifest {info['sha256']}")
        total_rows += rows
    if total_rows != manifest["count"]:
        raise DatasetValidationError(f"manifest count mismatch: {total_rows} != {manifest['count']}")
    return manifest
