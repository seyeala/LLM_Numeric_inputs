import pytest
import torch
from conftest import DummyCausalLM

from llm_numeric_inputs.experiments.tiny_model import TinyTokenizer
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.wrapper import NumericLMWrapper


def mixed_wrapper(hidden_size=4, n_positions=8):
    model = DummyCausalLM(hidden_size=hidden_size, n_positions=n_positions, vocab_size=128)
    return NumericLMWrapper(
        "dummy",
        mode=InputMode.MIXED_TO_NUMERIC,
        model=model,
        tokenizer=TinyTokenizer(),
    )


def test_mixed_forward_one_marker_returns_batch_shape():
    wrapper = mixed_wrapper()
    output = wrapper({"input_text": ["value $$1.5&&", "value $$2.5&&"]})
    assert tuple(output.shape) == (2, 1)


def test_mixed_forward_batch_size_one_preserves_batch():
    wrapper = mixed_wrapper()
    output = wrapper({"input_text": "only $$3.25&&"})
    assert tuple(output.shape) == (1, 1)


def test_mixed_forward_multiple_markers_uses_valid_hidden_size():
    wrapper = mixed_wrapper(hidden_size=5, n_positions=8)
    output = wrapper({"input_text": ["a $$1&& b $$2&& c", "plain text only"]})
    assert tuple(output.shape) == (2, 1)
    assert wrapper.model.last_inputs_embeds_shape[0] == 2
    assert wrapper.model.last_inputs_embeds_shape[2] == 5


def test_mixed_forward_supports_marker_only_sequence():
    wrapper = mixed_wrapper()
    output = wrapper({"input_text": ["$$9.5&&"]})
    assert tuple(output.shape) == (1, 1)
    assert wrapper.model.last_inputs_embeds_shape[1] == 1


def test_mixed_forward_zero_marker_text_sequence():
    wrapper = mixed_wrapper()
    output = wrapper({"input_text": ["plain text without numeric marker"]})
    assert tuple(output.shape) == (1, 1)


@pytest.mark.parametrize("bad", ["bad $$x&&", "bad $$12", "bad 12&&", "$$&&"])
def test_mixed_forward_rejects_malformed_markers(bad):
    wrapper = mixed_wrapper()
    with pytest.raises(ValueError):
        wrapper({"input_text": [bad]})


def test_mixed_forward_rejects_empty_batch():
    wrapper = mixed_wrapper()
    with pytest.raises(ValueError, match="must not be empty"):
        wrapper({"input_text": []})


def test_mixed_training_step_updates_projections_not_transformer():
    wrapper = mixed_wrapper()
    wrapper.configure_trainable_layers(True, True, False)
    before_transformer = {key: value.detach().clone() for key, value in wrapper.model.state_dict().items()}
    before_projection = wrapper.input_projection.weight.detach().clone()
    optimizer = torch.optim.Adam([param for param in wrapper.parameters() if param.requires_grad], lr=0.01)
    output = wrapper({"input_text": ["x $$1&&", "y $$2&&"]})
    loss = torch.nn.MSELoss()(output, torch.tensor([[1.0], [2.0]]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before_projection, wrapper.input_projection.weight.detach())
    for key, value in wrapper.model.state_dict().items():
        assert torch.equal(before_transformer[key], value.detach())


def test_mixed_forward_requires_tokenizer_for_text_segments():
    wrapper = NumericLMWrapper(
        "dummy",
        mode=InputMode.MIXED_TO_NUMERIC,
        model=DummyCausalLM(),
        tokenizer=None,
    )
    with pytest.raises(ValueError, match="require a tokenizer"):
        wrapper({"input_text": ["text $$1&&"]})
