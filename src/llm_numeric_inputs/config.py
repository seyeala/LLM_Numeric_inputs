"""Shared configuration and CLI helpers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_numeric_inputs.modes import NumericSequenceStrategy


@dataclass(slots=True)
class TrainConfig:
    model_name: str = "openai-community/gpt2"
    device: str = "auto"
    num_batches: int = 50
    batch_size: int = 8
    lr: float = 0.003
    num_epochs: int = 60
    min_val: float = 0.0
    max_val: float = 100.0
    shl: bool = True
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    model_path_load: str | None = None
    model_path_save: str = "./chk/trained_numeric_lm_stage1.pth"
    seed: int = 42
    deterministic: bool = False
    numeric_sequence_strategy: NumericSequenceStrategy = NumericSequenceStrategy.SINGLE_TOKEN

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["numeric_sequence_strategy"] = str(self.numeric_sequence_strategy)
        return data


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _normalize_optional_path(value: Any) -> str | None:
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def train_config_from_mapping(mapping: dict[str, Any] | None) -> TrainConfig:
    data = dict(mapping or {})
    if "model_path_load" in data:
        data["model_path_load"] = _normalize_optional_path(data["model_path_load"])
    if "numeric_sequence_strategy" in data:
        data["numeric_sequence_strategy"] = NumericSequenceStrategy(data["numeric_sequence_strategy"])
    allowed = set(TrainConfig.__dataclass_fields__)
    filtered = {key: value for key, value in data.items() if key in allowed}
    config = TrainConfig(**filtered)
    validate_config(config)
    return config


def apply_cli_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    data = config.to_dict()
    for key in data:
        value = getattr(args, key, None)
        if value is not None:
            data[key] = value
    if "model_path_load" in data:
        data["model_path_load"] = _normalize_optional_path(data["model_path_load"])
    return train_config_from_mapping(data)


def load_train_config(config_path: str | Path, args: argparse.Namespace | None = None) -> TrainConfig:
    config = train_config_from_mapping(load_config(config_path))
    return apply_cli_overrides(config, args) if args is not None else config


def validate_config(config: TrainConfig) -> None:
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if config.num_batches <= 0:
        raise ValueError("num_batches must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    if config.min_val >= config.max_val:
        raise ValueError("min_val must be smaller than max_val.")
    if config.scheduler_step_size <= 0:
        raise ValueError("scheduler_step_size must be positive.")
    if not 0 < config.scheduler_gamma <= 1:
        raise ValueError("scheduler_gamma must be in (0, 1].")
    if not config.model_path_save:
        raise ValueError("model_path_save must be set when training scripts save a checkpoint.")


def add_common_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--num_batches", type=int, help="Number of batches per epoch.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--min_val", type=float, help="Minimum generated numeric value.")
    parser.add_argument("--max_val", type=float, help="Maximum generated numeric value.")
    parser.add_argument("--model_name", type=str, help="Hugging Face model name or local path.")
    parser.add_argument("--model_path_load", type=str, help="Checkpoint path to load before training.")
    parser.add_argument("--model_path_save", type=str, help="Checkpoint path to save after training.")
    parser.add_argument("--device", type=str, help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--deterministic", type=str2bool, nargs="?", const=True, help="Use deterministic algorithms.")
    parser.add_argument("--shl", type=str2bool, nargs="?", const=True, help="Enable StepLR scheduler.")
    parser.add_argument("--scheduler_step_size", type=int, help="StepLR step size.")
    parser.add_argument("--scheduler_gamma", type=float, help="StepLR gamma.")
    parser.add_argument("--numeric_sequence_strategy", type=str, choices=[x.value for x in NumericSequenceStrategy])
    return parser
