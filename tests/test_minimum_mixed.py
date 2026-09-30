import torch

from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.experiments.minimum_alignment import (
    MinimumExperimentConfig,
    build_minimum_model,
    load_minimum_experiment_config,
    train_minimum_alignment,
)
from llm_numeric_inputs.experiments.minimum_data import read_minimum_split
from llm_numeric_inputs.experiments.minimum_evaluation import evaluate_minimum_alignment
from llm_numeric_inputs.experiments.tiny_model import TINY_MODEL_NAME


def make_manifest(tmp_path):
    return generate_dataset("minimum", 24, 909, tmp_path / "minimum", validate=True)


def test_read_minimum_split_shapes(tmp_path):
    manifest = make_manifest(tmp_path)
    texts, targets, loaded = read_minimum_split(manifest, "train")
    assert texts
    assert targets.ndim == 2
    assert targets.shape[1] == 1
    assert loaded["task"] == "minimum"


def test_evaluate_minimum_alignment_returns_metrics(tmp_path):
    manifest = make_manifest(tmp_path)
    config = MinimumExperimentConfig(train=TrainConfig(model_name=TINY_MODEL_NAME, batch_size=4, num_epochs=1))
    model = build_minimum_model(config, torch.device("cpu"))
    metrics = evaluate_minimum_alignment(model, manifest, "test", 4, torch.device("cpu"))
    assert metrics["split"] == "test"
    assert metrics["task"] == "minimum"
    assert metrics["count"] == metrics["dataset_count"]
    assert "mae" in metrics


def test_minimum_alignment_smoke_run(tmp_path):
    manifest = make_manifest(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
validation_split: validation
test_split: test
run_id: test_minimum
train:
  model_name: "__tiny_numeric_lm__"
  device: "cpu"
  batch_size: 4
  lr: 0.003
  num_epochs: 1
  num_batches: 1
  min_val: 0
  max_val: 100
  seed: 909
  deterministic: false
  shl: false
  model_path_save: "./chk/test_minimum.pth"
""".strip(),
        encoding="utf-8",
    )
    run_dir = train_minimum_alignment(load_minimum_experiment_config(config_path), manifest, tmp_path / "runs")
    assert (run_dir / "checkpoint.pth").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "run_manifest.json").exists()
