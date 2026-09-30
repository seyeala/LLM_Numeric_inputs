import pytest

from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.experiments.text_data import read_text_split, tokenize_text_batch
from llm_numeric_inputs.experiments.tiny_model import TinyTokenizer


def test_read_text_split_shapes(tmp_path):
    manifest = generate_dataset("text_to_numeric", 12, 707, tmp_path / "text", validate=True)
    texts, targets, loaded = read_text_split(manifest, "train")
    assert texts
    assert targets.ndim == 2
    assert targets.shape[1] == 1
    assert loaded["task"] == "text_to_numeric"


def test_read_text_split_rejects_wrong_task(tmp_path):
    manifest = generate_dataset("numeric_to_numeric", 8, 707, tmp_path / "numeric", validate=True)
    with pytest.raises(ValueError, match="text_to_numeric"):
        read_text_split(manifest, "train")


def test_tokenize_text_batch_moves_tensors_to_device():
    tokenized = tokenize_text_batch(TinyTokenizer(), ["$$1.0&&", "$$22.0&&"], device="cpu")
    assert set(tokenized) == {"input_ids", "attention_mask"}
    assert tokenized["input_ids"].shape[0] == 2
