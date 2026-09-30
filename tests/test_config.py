import argparse

import pytest

from llm_numeric_inputs.config import TrainConfig, apply_cli_overrides, load_train_config, train_config_from_mapping
from llm_numeric_inputs.modes import NumericSequenceStrategy


def test_train_config_defaults_are_valid():
    config = TrainConfig()
    assert config.batch_size == 8
    assert config.numeric_sequence_strategy == NumericSequenceStrategy.SINGLE_TOKEN


def test_yaml_loading_and_cli_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 2\nlr: 0.01\nmodel_path_load: ''\n", encoding="utf-8")
    args = argparse.Namespace(batch_size=4, lr=None, model_path_load=None, model_path_save="out/checkpoint.pth")
    config = load_train_config(config_path, args)
    assert config.batch_size == 4
    assert config.lr == 0.01
    assert config.model_path_load is None
    assert config.model_path_save == "out/checkpoint.pth"


@pytest.mark.parametrize("value", [None, "", "null", "None"])
def test_model_path_load_normalization(value):
    config = train_config_from_mapping({"model_path_load": value})
    assert config.model_path_load is None


def test_numeric_sequence_strategy_parsing():
    config = train_config_from_mapping({"numeric_sequence_strategy": "repeat_n_positions"})
    assert config.numeric_sequence_strategy == NumericSequenceStrategy.REPEAT_N_POSITIONS


@pytest.mark.parametrize(
    "updates",
    [
        {"num_epochs": 0},
        {"num_batches": 0},
        {"batch_size": 0},
        {"lr": 0},
        {"min_val": 10, "max_val": 1},
        {"scheduler_step_size": 0},
        {"scheduler_gamma": 0},
        {"scheduler_gamma": 1.1},
        {"model_path_save": ""},
    ],
)
def test_invalid_config_values_raise(updates):
    with pytest.raises(ValueError):
        train_config_from_mapping(updates)


def test_unknown_config_keys_are_ignored_for_now():
    config = train_config_from_mapping({"unknown_key": "ignored"})
    assert not hasattr(config, "unknown_key")


def test_apply_cli_overrides_keeps_unspecified_values():
    config = TrainConfig(batch_size=3, lr=0.02)
    args = argparse.Namespace(batch_size=None, lr=0.01)
    updated = apply_cli_overrides(config, args)
    assert updated.batch_size == 3
    assert updated.lr == 0.01
