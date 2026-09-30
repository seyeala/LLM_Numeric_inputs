import torch
from conftest import DummyCausalLM, DummyTokenizer

from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.wrapper import NumericLMWrapper


def make_wrapper():
    return NumericLMWrapper(
        "dummy",
        mode=InputMode.NUMERIC_TO_NUMERIC,
        model=DummyCausalLM(),
        tokenizer=DummyTokenizer(),
    )


def clone_params(module):
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def test_configure_trainable_layers_flags():
    wrapper = make_wrapper()
    wrapper.configure_trainable_layers(False, True, False)
    assert not any(param.requires_grad for param in wrapper.input_projection.parameters())
    assert all(param.requires_grad for param in wrapper.output_projection.parameters())
    assert not any(param.requires_grad for param in wrapper.model.parameters())


def test_frozen_transformer_weights_do_not_change_but_projection_does():
    wrapper = make_wrapper()
    wrapper.configure_trainable_layers(True, True, False)
    transformer_before = clone_params(wrapper.model)
    projection_before = clone_params(wrapper.input_projection)

    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, wrapper.parameters()), lr=0.1)
    loss = wrapper(torch.tensor([[1.0], [2.0], [3.0]])).pow(2).mean()
    loss.backward()
    optimizer.step()

    for name, before in transformer_before.items():
        assert torch.equal(before, dict(wrapper.model.named_parameters())[name])
    projection_after = dict(wrapper.input_projection.named_parameters())
    assert any(not torch.equal(before, projection_after[name]) for name, before in projection_before.items())
