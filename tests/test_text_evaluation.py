import torch

from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.experiments.text_alignment import TextExperimentConfig, build_text_model
from llm_numeric_inputs.experiments.text_evaluation import evaluate_text_alignment
from llm_numeric_inputs.experiments.tiny_model import TINY_MODEL_NAME


def test_evaluate_text_alignment_returns_metrics(tmp_path):
    manifest = generate_dataset("text_to_numeric", 12, 707, tmp_path / "text", validate=True)
    config = TextExperimentConfig(train=TrainConfig(model_name=TINY_MODEL_NAME, batch_size=4, num_epochs=1))
    model = build_text_model(config, torch.device("cpu"))
    metrics = evaluate_text_alignment(model, model.tokenizer, manifest, "test", 4, torch.device("cpu"))
    assert metrics["split"] == "test"
    assert metrics["count"] == metrics["dataset_count"]
    assert metrics["task"] == "text_to_numeric"
    assert "mae" in metrics
