import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.wrapper import NumericLMWrapper

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class DummyBatch(dict):
    def to(self, device):
        return DummyBatch({key: value.to(device) for key, value in self.items()})


class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        del return_tensors, padding, truncation
        max_len = max(len(text) for text in texts)
        ids = []
        masks = []
        for text in texts:
            row = [(ord(ch) % 15) + 1 for ch in text]
            mask = [1] * len(row)
            row = row + [0] * (max_len - len(row))
            mask = mask + [0] * (max_len - len(mask))
            ids.append(row)
            masks.append(mask)
        return DummyBatch({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)})


class DummyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 4, n_positions: int = 6, vocab_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, n_positions=n_positions)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.last_inputs_embeds_shape = None

    def forward(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, return_dict=True, output_hidden_states=True
    ):
        del attention_mask, return_dict, output_hidden_states
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
        hidden = self.linear(inputs_embeds)
        return SimpleNamespace(logits=hidden, hidden_states=(hidden,))


class PositionCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 1, n_positions: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, n_positions=n_positions)
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, return_dict=True, output_hidden_states=True
    ):
        del inputs_embeds, attention_mask, return_dict, output_hidden_states
        batch, seq_len = input_ids.shape
        positions = torch.arange(1, seq_len + 1, dtype=torch.float32, device=input_ids.device)
        hidden = positions.view(1, seq_len, 1).expand(batch, seq_len, 1) + self.offset
        return SimpleNamespace(logits=hidden, hidden_states=(hidden,))


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()


@pytest.fixture
def numeric_wrapper():
    return NumericLMWrapper(
        "dummy",
        mode=InputMode.NUMERIC_TO_NUMERIC,
        model=DummyCausalLM(),
        tokenizer=DummyTokenizer(),
    )


@pytest.fixture
def text_wrapper():
    return NumericLMWrapper(
        "dummy",
        mode=InputMode.TEXT_TO_NUMERIC,
        model=DummyCausalLM(),
        tokenizer=DummyTokenizer(),
    )
