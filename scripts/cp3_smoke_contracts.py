"""CP3 smoke checks for wrapper contracts and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from llm_numeric_inputs.checkpointing import load_checkpoint_into_model, save_checkpoint
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.pooling import last_valid_token_pool
from llm_numeric_inputs.wrapper import NumericLMWrapper


class DummyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 4, n_positions: int = 6, vocab_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, n_positions=n_positions)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, return_dict=True, output_hidden_states=True
    ):
        del attention_mask, return_dict, output_hidden_states
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        hidden = self.linear(inputs_embeds)
        return SimpleNamespace(logits=hidden, hidden_states=(hidden,))


def assert_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise AssertionError(f"Expected shape {shape}, got {tuple(tensor.shape)}")


def main() -> None:
    numeric = NumericLMWrapper("dummy", mode=InputMode.NUMERIC_TO_NUMERIC, model=DummyCausalLM(), tokenizer=None)
    y_numeric = numeric(torch.tensor([[1.0], [2.0], [3.0]]))
    assert_shape(y_numeric, (3, 1))

    text = NumericLMWrapper("dummy", mode=InputMode.TEXT_TO_NUMERIC, model=DummyCausalLM(), tokenizer=None)
    tokenized = {
        "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }
    y_text = text(tokenized)
    assert_shape(y_text, (2, 1))

    hidden = torch.tensor([[[1.0], [2.0], [99.0]], [[4.0], [5.0], [6.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = last_valid_token_pool(hidden, mask)
    if pooled.tolist() != [[2.0], [6.0]]:
        raise AssertionError(f"Unexpected pooled values: {pooled.tolist()}")

    path = Path("chk/cp3_smoke_checkpoint.pth")
    save_checkpoint(numeric, path, InputMode.NUMERIC_TO_NUMERIC)
    reloaded = NumericLMWrapper("dummy", mode=InputMode.NUMERIC_TO_NUMERIC, model=DummyCausalLM(), tokenizer=None)
    load_checkpoint_into_model(reloaded, path)
    for key, value in numeric.state_dict().items():
        if key.startswith(("input_projection.", "output_projection.")) and not torch.equal(
            value, reloaded.state_dict()[key]
        ):
            raise AssertionError(f"Projection key did not round-trip: {key}")

    legacy_path = Path("chk/cp3_legacy_state_dict.pth")
    torch.save(numeric.state_dict(), legacy_path)
    load_checkpoint_into_model(reloaded, legacy_path)
    print("CP3 smoke contracts: ok")


if __name__ == "__main__":
    main()
