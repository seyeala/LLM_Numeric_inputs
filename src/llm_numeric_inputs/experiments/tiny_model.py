"""Tiny local causal LM used for offline experiment smoke tests."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

TINY_MODEL_NAME = "__tiny_numeric_lm__"


class TinyCausalLM(nn.Module):
    """Small causal-LM-like module returning hidden states for wrapper tests/runs."""

    def __init__(self, hidden_size: int = 8, n_positions: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, n_positions=n_positions)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, return_dict=True, output_hidden_states=True
    ):
        del input_ids, attention_mask, return_dict, output_hidden_states
        if inputs_embeds is None:
            raise ValueError("TinyCausalLM requires inputs_embeds.")
        hidden = self.linear(inputs_embeds)
        return SimpleNamespace(logits=hidden, hidden_states=(hidden,))
