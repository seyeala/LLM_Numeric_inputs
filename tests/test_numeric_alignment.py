import torch

from llm_numeric_inputs.checkpointing import load_checkpoint
from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.experiments.numeric_alignment import (
    load_numeric_experiment_config,
    train_numeric_alignment,
)


def test_numeric_alignment_smoke_run(tmp_path):
    manifest = generate_dataset("numeric_to_numeric", 24, 606, tmp_path / "data", validate=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
validation_split: validation
test_split: test
run_id: test_run
train:
  model_name: "__tiny_numeric_lm__"
  device: "cpu"
  batch_size: 4
  lr: 0.003
  num_epochs: 1
  num_batches: 1
  min_val: 0
  max_val: 100
  seed: 606
  deterministic: false
  shl: false
  model_path_save: "./chk/test.pth"
""".strip(),
        encoding="utf-8",
    )
    run_dir = train_numeric_alignment(load_numeric_experiment_config(config_path), manifest, tmp_path / "runs")
    assert (run_dir / "checkpoint.pth").exists()
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "environment_report.json").exists()
    assert (run_dir / "epoch_metrics.json").exists()
    assert (run_dir / "test_metrics.json").exists()
    payload = load_checkpoint(run_dir / "checkpoint.pth", map_location=torch.device("cpu"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["cp"] == "CP6"
