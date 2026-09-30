import pytest
import torch
from conftest import DummyCausalLM, DummyTokenizer, PositionCausalLM

from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.wrapper import NumericLMWrapper


def make_text_wrapper(model=None):
    return NumericLMWrapper(
        "dummy",
        mode=InputMode.TEXT_TO_NUMERIC,
        model=model or DummyCausalLM(),
        tokenizer=DummyTokenizer(),
    )


def test_text_forward_shape():
    wrapper = make_text_wrapper()
    tokenized = {
        "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }
    assert tuple(wrapper(tokenized).shape) == (2, 1)


def test_text_forward_rejects_non_dict():
    wrapper = make_text_wrapper()
    with pytest.raises(TypeError):
        wrapper(torch.ones(2, 3))


def test_text_forward_requires_input_ids():
    wrapper = make_text_wrapper()
    with pytest.raises(ValueError, match="input_ids"):
        wrapper({"attention_mask": torch.ones(1, 2, dtype=torch.long)})


def test_text_forward_adds_missing_attention_mask():
    wrapper = make_text_wrapper()
    output = wrapper({"input_ids": torch.tensor([[1, 2, 3]])})
    assert tuple(output.shape) == (1, 1)


def test_text_forward_uses_attention_mask_for_pooling():
    wrapper = make_text_wrapper(model=PositionCausalLM())
    with torch.no_grad():
        wrapper.output_projection.weight.fill_(1.0)
        wrapper.output_projection.bias.zero_()
    tokenized = {
        "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }
    output = wrapper(tokenized)
    assert torch.allclose(output, torch.tensor([[2.0], [3.0]]))
