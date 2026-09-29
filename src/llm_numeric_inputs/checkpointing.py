"""Versioned checkpoint helpers."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.modes import InputMode

SCHEMA_VERSION = 1
PROJECTION_PREFIXES = ("input_projection.", "output_projection.")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def projection_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return only projection-layer weights from a wrapper state dict."""
    return {
        key: value.detach().cpu() for key, value in model.state_dict().items() if key.startswith(PROJECTION_PREFIXES)
    }


def checkpoint_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    mode: InputMode | str,
    train_config: TrainConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a compact projection-only checkpoint with metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": str(mode),
        "model_name": getattr(model, "model_name", None),
        "numeric_sequence_strategy": str(getattr(model, "numeric_sequence_strategy", "")),
        "train_config": train_config.to_dict() if train_config else None,
        "projection_state_dict": projection_state_dict(model),
        "metadata": checkpoint_metadata(metadata),
    }
    torch.save(payload, output_path)


def _map_location(model: torch.nn.Module | None = None) -> torch.device | str:
    if model is None:
        return "cpu"
    return getattr(model, "device", torch.device("cpu"))


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a trusted local project checkpoint with metadata preserved."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint_into_model(model: torch.nn.Module, path: str | Path, strict: bool = False):
    """Load either a CP3 schema checkpoint or a legacy full state_dict."""
    payload = load_checkpoint(path, map_location=_map_location(model))
    if isinstance(payload, dict) and payload.get("schema_version") == SCHEMA_VERSION:
        state = payload["projection_state_dict"]
    elif isinstance(payload, dict):
        state = {
            key: value
            for key, value in payload.items()
            if key.startswith(PROJECTION_PREFIXES) or key.startswith("model.")
        }
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return model.load_state_dict(state, strict=strict)


def load_specific_weights(model: torch.nn.Module, load_path: str | Path):
    """Compatibility wrapper used by pre-CP3 scripts."""
    result = load_checkpoint_into_model(model, load_path, strict=False)
    if getattr(result, "unexpected_keys", None):
        print(f"Unexpected checkpoint keys ignored: {result.unexpected_keys}")
    return model
