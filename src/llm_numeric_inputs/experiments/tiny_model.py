"""Tiny local causal LM and tokenizer used for offline experiment smoke tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

TINY_MODEL_NAME = "__tiny_numeric_lm__"


class TinyBatch(dict):
    """Small tokenizer-batch dict with a Hugging-Face-like .to method."""

    def to(self, device: torch.device | str):
        return TinyBatch({key: value.to(device) for key, value in self.items()})


class TinyTokenizer:
    """Deterministic character tokenizer for offline text experiment tests."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def _encode(self, text: str) -> list[int]:
        return [min(ord(ch), self.vocab_size - 1) for ch in text]

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        del return_tensors, padding, truncation
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode(text) or [1] for text in texts]
        width = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            pad = width - len(row)
            input_ids.append(row + [0] * pad)
            attention_mask.append([1] * len(row) + [0] * pad)
        return TinyBatch(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class TinyCausalLM(nn.Module):
    """Small causal-LM-like module returning hidden states for wrapper tests/runs."""

    def __init__(self, hidden_size: int = 8, n_positions: int = 4, vocab_size: int = 128):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, n_positions=n_positions)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        return_dict=True,
        output_hidden_states=True,
    ):
        del attention_mask, return_dict, output_hidden_states
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("TinyCausalLM requires input_ids or inputs_embeds.")
            inputs_embeds = self.embedding(input_ids)
        hidden = self.linear(inputs_embeds)
        return SimpleNamespace(logits=hidden, hidden_states=(hidden,))
