"""Dataset task names and schema constants."""

from __future__ import annotations

from enum import StrEnum


class TaskName(StrEnum):
    """Supported deterministic dataset tasks."""

    NUMERIC_TO_NUMERIC = "numeric_to_numeric"
    TEXT_TO_NUMERIC = "text_to_numeric"
    MINIMUM = "minimum"


SPLIT_NAMES = ("train", "validation", "test")
SCHEMA_VERSION = 1
REQUIRED_RECORD_KEYS = ("input", "output")
