from pathlib import Path

import torch

from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.experiments.evaluation import evaluate_numeric_alignment
from llm_numeric_inputs.experiments.numeric_alignment import NumericExperimentConfig, build_numeric_model
from llm_numeric_inputs.experiments.numeric_data import read_numeric_split
from llm_numeric_inputs.experiments.tiny_model import TINY_MODEL_NAME


def make_manifest(tmp_path: Path) -> Path:
    return generate_dataset("numeric_to_numeric", 12, 123, tmp_path / "numeric", validate=True)


def test_read_numeric_split_shapes(tmp_path):
    manifest = make_manifest(tmp_path)
    inputs, targets, loaded = read_numeric_split(manifest, "train")
    assert inputs.ndim == 2
    assert targets.ndim == 2
    assert inputs.shape[1] == 1
    assert loaded["task"] == "numeric_to_numeric"


def test_evaluate_numeric_alignment_returns_metrics(tmp_path):
    manifest = make_manifest(tmp_path)
    config = NumericExperimentConfig(train=TrainConfig(model_name=TINY_MODEL_NAME, batch_size=4, num_epochs=1))
    model = build_numeric_model(config, torch.device("cpu"))
    metrics = evaluate_numeric_alignment(model, manifest, "test", 4, torch.device("cpu"))
    assert metrics["split"] == "test"
    assert metrics["count"] == metrics["dataset_count"]
    assert "mae" in metrics
