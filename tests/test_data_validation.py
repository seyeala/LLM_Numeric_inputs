import json

import pytest

from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.data.schemas import TaskName
from llm_numeric_inputs.data.validation import DatasetValidationError, validate_jsonl_file, validate_manifest


def test_validate_jsonl_file_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="invalid JSON"):
        validate_jsonl_file(path, TaskName.NUMERIC_TO_NUMERIC)


def test_validate_jsonl_file_rejects_wrong_numeric_label(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"input": "1.0", "output": "2.0"}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="mismatch"):
        validate_jsonl_file(path, TaskName.NUMERIC_TO_NUMERIC)


def test_validate_jsonl_file_rejects_wrong_minimum_label(tmp_path):
    path = tmp_path / "bad.jsonl"
    record = {"input": "Among $$3.00&& and $$2.00&&, find the lowest number.", "output": "3.00"}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="does not equal"):
        validate_jsonl_file(path, TaskName.MINIMUM)


def test_validate_manifest_rejects_hash_mismatch(tmp_path):
    manifest_path = generate_dataset(TaskName.NUMERIC_TO_NUMERIC, count=10, seed=42, output_dir=tmp_path)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + '{"input":"1","output":"1"}\n', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="row count"):
        validate_manifest(manifest_path)
