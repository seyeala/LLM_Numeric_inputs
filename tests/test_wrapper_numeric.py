import pytest
import torch
from conftest import DummyCausalLM, DummyTokenizer

from llm_numeric_inputs.modes import InputMode, NumericSequenceStrategy
from llm_numeric_inputs.wrapper import NumericLMWrapper


def make_wrapper(strategy=NumericSequenceStrategy.SINGLE_TOKEN):
    return NumericLMWrapper(
        "dummy",
        mode=InputMode.NUMERIC_TO_NUMERIC,
        model=DummyCausalLM(n_positions=5),
        tokenizer=DummyTokenizer(),
        numeric_sequence_strategy=strategy,
    )


def test_numeric_forward_shape():
    wrapper = make_wrapper()
    output = wrapper(torch.tensor([[1.0], [2.0]]))
    assert tuple(output.shape) == (2, 1)


def test_numeric_forward_rejects_non_tensor():
    wrapper = make_wrapper()
    with pytest.raises(TypeError):
        wrapper([1.0, 2.0])


@pytest.mark.parametrize("bad_shape", [torch.ones(2), torch.ones(2, 2), torch.ones(1, 1, 1)])
def test_numeric_forward_rejects_invalid_shapes(bad_shape):
    wrapper = make_wrapper()
    with pytest.raises(ValueError, match="shape"):
        wrapper(bad_shape)


def test_single_token_numeric_strategy_uses_sequence_length_one():
    wrapper = make_wrapper(NumericSequenceStrategy.SINGLE_TOKEN)
    wrapper(torch.tensor([[1.0], [2.0]]))
    assert wrapper.model.last_inputs_embeds_shape == (2, 1, 4)


def test_repeat_strategy_uses_configured_context_length():
    wrapper = make_wrapper(NumericSequenceStrategy.REPEAT_N_POSITIONS)
    wrapper(torch.tensor([[1.0], [2.0]]))
    assert wrapper.model.last_inputs_embeds_shape == (2, 5, 4)


def test_numeric_gradients_reach_projection_layers():
    wrapper = make_wrapper()
    loss = wrapper(torch.tensor([[1.0], [2.0]])).sum()
    loss.backward()
    assert wrapper.input_projection.weight.grad is not None
    assert wrapper.output_projection.weight.grad is not None
