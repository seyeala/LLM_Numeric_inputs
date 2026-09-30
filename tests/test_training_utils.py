import pytest
import torch
from conftest import DummyCausalLM, DummyTokenizer

from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.training_utils import (
    generate_numeric_batch,
    generate_text_numeric_batch,
    train_projection_epochs,
)
from llm_numeric_inputs.wrapper import NumericLMWrapper


def make_wrapper():
    wrapper = NumericLMWrapper(
        "dummy",
        mode=InputMode.NUMERIC_TO_NUMERIC,
        model=DummyCausalLM(),
        tokenizer=DummyTokenizer(),
    )
    wrapper.configure_trainable_layers(True, True, False)
    return wrapper


def test_generate_numeric_batch_shapes():
    config = TrainConfig(batch_size=3)
    inputs, targets = generate_numeric_batch(config, torch.device("cpu"))
    assert tuple(inputs.shape) == (3, 1)
    assert torch.equal(inputs, targets)


def test_generate_text_numeric_batch_shapes(dummy_tokenizer):
    config = TrainConfig(batch_size=2)
    tokenized, targets = generate_text_numeric_batch(config, torch.device("cpu"), dummy_tokenizer)
    assert tuple(targets.shape) == (2, 1)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert tokenized["input_ids"].shape[0] == 2


def test_train_projection_epochs_one_step_runs(capsys):
    config = TrainConfig(num_epochs=1, num_batches=1, batch_size=2, shl=False)
    wrapper = make_wrapper()
    train_projection_epochs(wrapper, config, lambda: generate_numeric_batch(config, torch.device("cpu")))
    assert "Average Loss" in capsys.readouterr().out


def test_train_projection_epochs_raises_when_no_trainable_parameters():
    config = TrainConfig(num_epochs=1, num_batches=1, batch_size=2, shl=False)
    wrapper = make_wrapper()
    wrapper.configure_trainable_layers(False, False, False)
    with pytest.raises(ValueError, match="No trainable parameters"):
        train_projection_epochs(wrapper, config, lambda: generate_numeric_batch(config, torch.device("cpu")))
