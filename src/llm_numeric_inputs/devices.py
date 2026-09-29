"""Device selection and CUDA utility helpers."""

from __future__ import annotations

import torch


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    """Resolve a requested device string into a torch.device with clear errors."""
    if isinstance(requested, torch.device):
        return requested
    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    value = str(requested).lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device {requested!r} was requested, but CUDA is not available.")
        device = torch.device(value)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is out of range.")
        return device
    raise ValueError(f"Unsupported device specifier: {requested!r}")


def clear_cuda_memory() -> None:
    """Clear CUDA cache when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA memory cache.")
    else:
        print("CUDA is not available; no CUDA cache to clear.")


def print_cuda_memory() -> None:
    """Print CUDA memory statistics when CUDA is available."""
    if not torch.cuda.is_available():
        print("CUDA is not available; memory stats are unavailable.")
        return
    print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6} MB")
    torch.cuda.reset_peak_memory_stats()
