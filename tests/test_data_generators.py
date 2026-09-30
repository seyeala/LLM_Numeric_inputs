import importlib.util
from pathlib import Path

from llm_numeric_inputs.data.generators import generate_records
from llm_numeric_inputs.data.schemas import TaskName


def test_legacy_generator_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module_path = Path(__file__).resolve().parents[1] / "dataset" / "text_numeric_generator_for_llm.py"
    spec = importlib.util.spec_from_file_location("legacy_generator_no_side_effects", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(tmp_path.iterdir()) == []


def test_same_seed_produces_identical_records():
    first = list(generate_records(TaskName.MINIMUM, count=10, seed=123))
    second = list(generate_records(TaskName.MINIMUM, count=10, seed=123))
    assert first == second


def test_different_seed_changes_records():
    first = list(generate_records(TaskName.TEXT_TO_NUMERIC, count=10, seed=123))
    second = list(generate_records(TaskName.TEXT_TO_NUMERIC, count=10, seed=124))
    assert first != second


def test_numeric_to_numeric_labels_are_identity():
    records = list(generate_records(TaskName.NUMERIC_TO_NUMERIC, count=10, seed=7))
    assert all(record["input"] == record["output"] for record in records)


def test_text_to_numeric_labels_match_marker():
    records = list(generate_records(TaskName.TEXT_TO_NUMERIC, count=10, seed=7))
    for record in records:
        assert record["input"] == f"$${record['output']}&&"


def test_minimum_labels_match_displayed_values():
    records = list(generate_records(TaskName.MINIMUM, count=10, seed=7))
    for record in records:
        values = [float(value.split("&&")[0]) for value in record["input"].split("$$")[1:]]
        assert float(record["output"]) == min(values)
