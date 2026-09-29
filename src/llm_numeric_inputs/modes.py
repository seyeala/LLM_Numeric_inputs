"""Mode declarations for supported numeric-input tasks."""

from __future__ import annotations

from enum import StrEnum


class InputMode(StrEnum):
    """Supported wrapper input modes."""

    NUMERIC_TO_NUMERIC = "numeric_to_numeric"
    TEXT_TO_NUMERIC = "text_to_numeric"
    MIXED_TO_NUMERIC = "mixed_to_numeric"


class NumericSequenceStrategy(StrEnum):
    """Ways to expose a scalar numeric projection to the transformer."""

    SINGLE_TOKEN = "single_token"
    REPEAT_N_POSITIONS = "repeat_n_positions"
