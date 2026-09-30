import json

from llm_numeric_inputs.experiments.benchmarks import collect_and_write, collect_runs, summarize_run


def make_run(root, name="run1"):
    run = root / name
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "checkpoint": str(run / "checkpoint.pth"),
                "dataset_manifest": "data/manifest.json",
                "dataset_task": "numeric_to_numeric",
                "dataset_seed": 1,
            }
        ),
        encoding="utf-8",
    )
    (run / "test_metrics.json").write_text(json.dumps({"split": "test", "count": 2, "mae": 1.5}), encoding="utf-8")
    (run / "resolved_config.json").write_text(
        json.dumps({"train": {"model_name": "dummy", "device": "cpu", "seed": 1, "batch_size": 2}}),
        encoding="utf-8",
    )
    (run / "environment_report.json").write_text(
        json.dumps({"python": "3.12", "packages": {"torch": "x"}}), encoding="utf-8"
    )
    return run


def test_summarize_run(tmp_path):
    run = make_run(tmp_path)
    summary = summarize_run(run)
    assert summary["run_id"] == "run1"
    assert summary["dataset_task"] == "numeric_to_numeric"
    assert summary["mae"] == 1.5


def test_collect_and_write(tmp_path):
    make_run(tmp_path / "runs", "run_a")
    rows = collect_runs(tmp_path / "runs")
    assert len(rows) == 1
    result = collect_and_write(tmp_path / "runs", tmp_path / "out")
    assert result["runs"] == 1
    assert (tmp_path / "out" / "benchmark_results.json").exists()
    assert (tmp_path / "out" / "benchmark_results.csv").exists()
