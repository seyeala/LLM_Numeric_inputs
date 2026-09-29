"""Core package for numeric projection experiments around causal language models."""

from llm_numeric_inputs.config import TrainConfig, apply_cli_overrides, load_config, validate_config
from llm_numeric_inputs.modes import InputMode, NumericSequenceStrategy
from llm_numeric_inputs.wrapper import NumericLMWrapper

__all__ = [
    "InputMode",
    "NumericLMWrapper",
    "NumericSequenceStrategy",
    "TrainConfig",
    "apply_cli_overrides",
    "load_config",
    "validate_config",
]
