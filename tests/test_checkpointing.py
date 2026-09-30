import pytest
import torch
from conftest import DummyCausalLM, DummyTokenizer

from llm_numeric_inputs.checkpointing import (
    load_checkpoint,
    load_checkpoint_into_model,
    projection_state_dict,
    save_checkpoint,
)
from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.wrapper import NumericLMWrapper


def make_wrapper():
    return NumericLMWrapper(
        "dummy-model",
        mode=InputMode.NUMERIC_TO_NUMERIC,
        model=DummyCausalLM(),
        tokenizer=DummyTokenizer(),
    )


def test_projection_state_dict_contains_only_projection_keys():
    state = projection_state_dict(make_wrapper())
    assert state
    assert all(key.startswith(("input_projection.", "output_projection.")) for key in state)


def test_save_checkpoint_schema_and_parent_creation(tmp_path):
    wrapper = make_wrapper()
    path = tmp_path / "nested" / "checkpoint.pth"
    config = TrainConfig(batch_size=2)
    save_checkpoint(wrapper, path, InputMode.NUMERIC_TO_NUMERIC, train_config=config)
    payload = load_checkpoint(path)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "numeric_to_numeric"
    assert payload["model_name"] == "dummy-model"
    assert payload["numeric_sequence_strategy"] == "single_token"
    assert payload["train_config"]["batch_size"] == 2
    assert "torch_version" in payload["metadata"]
    assert all(key.startswith(("input_projection.", "output_projection.")) for key in payload["projection_state_dict"])


def test_load_checkpoint_into_model_round_trips_projection_weights(tmp_path):
    wrapper = make_wrapper()
    with torch.no_grad():
        wrapper.input_projection.bias.fill_(3.14)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(wrapper, path, InputMode.NUMERIC_TO_NUMERIC)
    reloaded = make_wrapper()
    load_checkpoint_into_model(reloaded, path)
    assert torch.equal(wrapper.input_projection.bias, reloaded.input_projection.bias)


def test_load_legacy_state_dict(tmp_path):
    wrapper = make_wrapper()
    path = tmp_path / "legacy.pth"
    torch.save(wrapper.state_dict(), path)
    reloaded = make_wrapper()
    result = load_checkpoint_into_model(reloaded, path, strict=False)
    assert not result.unexpected_keys


def test_invalid_checkpoint_payload_type_raises(tmp_path):
    path = tmp_path / "invalid.pth"
    torch.save(["not", "a", "dict"], path)
    with pytest.raises(TypeError):
        load_checkpoint_into_model(make_wrapper(), path)
