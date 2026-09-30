import torch
import yaml

from llm_numeric_inputs.checkpointing import load_checkpoint, save_checkpoint
from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.experiments.text_alignment import (
    TextExperimentConfig,
    build_text_model,
    load_text_experiment_config,
    train_text_alignment,
)
from llm_numeric_inputs.experiments.tiny_model import TINY_MODEL_NAME
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.wrapper import NumericLMWrapper


def write_config(path, initialization="fresh", transfer_checkpoint=None):
    payload = {
        "validation_split": "validation",
        "test_split": "test",
        "run_id": "test_text_run",
        "initialization": initialization,
        "tiny_hidden_size": 8,
        "tiny_n_positions": 16,
        "train": {
            "model_name": TINY_MODEL_NAME,
            "device": "cpu",
            "batch_size": 4,
            "lr": 0.003,
            "num_epochs": 1,
            "num_batches": 1,
            "min_val": 0,
            "max_val": 100,
            "seed": 707,
            "deterministic": False,
            "shl": False,
            "model_path_save": "./chk/test_text.pth",
        },
    }
    if transfer_checkpoint:
        payload["transfer_checkpoint"] = str(transfer_checkpoint)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_text_alignment_smoke_run(tmp_path):
    manifest = generate_dataset("text_to_numeric", 24, 707, tmp_path / "data", validate=True)
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    run_dir = train_text_alignment(load_text_experiment_config(config_path), manifest, tmp_path / "runs")
    assert (run_dir / "checkpoint.pth").exists()
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "environment_report.json").exists()
    assert (run_dir / "epoch_metrics.json").exists()
    assert (run_dir / "test_metrics.json").exists()
    payload = load_checkpoint(run_dir / "checkpoint.pth", map_location=torch.device("cpu"))
    assert payload["schema_version"] == 1
    assert payload["mode"] == "text_to_numeric"
    assert payload["metadata"]["cp"] == "CP7"


def test_text_alignment_transfer_initialization_loads_projection(tmp_path):
    base = NumericLMWrapper(
        TINY_MODEL_NAME,
        mode=InputMode.TEXT_TO_NUMERIC,
        model=build_text_model(
            TextExperimentConfig(train=TrainConfig(model_name=TINY_MODEL_NAME)), torch.device("cpu")
        ).model,
        tokenizer=None,
    )
    with torch.no_grad():
        base.output_projection.bias.fill_(12.0)
    transfer = tmp_path / "transfer.pth"
    save_checkpoint(base, transfer, InputMode.TEXT_TO_NUMERIC, metadata={"test": "transfer"})

    config_path = tmp_path / "transfer_config.yaml"
    write_config(config_path, initialization="transfer", transfer_checkpoint=transfer)
    config = load_text_experiment_config(config_path)
    model = build_text_model(config, torch.device("cpu"))
    from llm_numeric_inputs.experiments.text_alignment import _load_transfer_if_requested

    _load_transfer_if_requested(model, config)
    assert torch.allclose(model.output_projection.bias, torch.tensor([12.0]))


def test_transfer_initialization_requires_checkpoint(tmp_path):
    config_path = tmp_path / "bad.yaml"
    write_config(config_path, initialization="transfer")
    import pytest

    with pytest.raises(ValueError, match="transfer"):
        load_text_experiment_config(config_path)
