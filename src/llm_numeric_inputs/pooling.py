"""Pooling functions for transformer hidden states."""

from __future__ import annotations

import torch


def last_valid_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the hidden state at the final non-padding token in each sample."""
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be rank 3 [batch, seq, hidden]; got {tuple(hidden_states.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank 2 [batch, seq]; got {tuple(attention_mask.shape)}")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_states and attention_mask batch/sequence dimensions must match.")
    lengths = attention_mask.to(torch.long).sum(dim=1) - 1
    if torch.any(lengths < 0):
        raise ValueError("attention_mask contains at least one sample with no valid tokens.")
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, lengths.to(hidden_states.device)]
