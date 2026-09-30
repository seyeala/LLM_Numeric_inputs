import json

from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.data.manifests import sha256_file
from llm_numeric_inputs.data.schemas import TaskName
from llm_numeric_inputs.data.validation import validate_manifest


def test_generate_dataset_writes_splits_and_manifest(tmp_path):
    manifest_path = generate_dataset(TaskName.NUMERIC_TO_NUMERIC, count=10, seed=42, output_dir=tmp_path)
    manifest = validate_manifest(manifest_path)
    assert manifest["schema_version"] == 1
    assert manifest["task"] == TaskName.NUMERIC_TO_NUMERIC.value
    assert manifest["counts"] == {"train": 8, "validation": 1, "test": 1}
    assert {info["rows"] for info in manifest["files"].values()} == {8, 1}


def test_same_seed_manifest_hashes_match(tmp_path):
    left = generate_dataset(TaskName.MINIMUM, count=12, seed=123, output_dir=tmp_path / "left")
    right = generate_dataset(TaskName.MINIMUM, count=12, seed=123, output_dir=tmp_path / "right")
    assert json.loads(left.read_text(encoding="utf-8")) == json.loads(right.read_text(encoding="utf-8"))


def test_different_seed_changes_file_hash(tmp_path):
    left = generate_dataset(TaskName.TEXT_TO_NUMERIC, count=12, seed=123, output_dir=tmp_path / "left")
    right = generate_dataset(TaskName.TEXT_TO_NUMERIC, count=12, seed=124, output_dir=tmp_path / "right")
    left_manifest = json.loads(left.read_text(encoding="utf-8"))
    right_manifest = json.loads(right.read_text(encoding="utf-8"))
    assert left_manifest["files"]["train.jsonl"]["sha256"] != right_manifest["files"]["train.jsonl"]["sha256"]


def test_manifest_sha_matches_file_content(tmp_path):
    manifest_path = generate_dataset(TaskName.NUMERIC_TO_NUMERIC, count=10, seed=42, output_dir=tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["train.jsonl"]["sha256"] == sha256_file(tmp_path / "train.jsonl")
