"""Benchmark/run aggregation for CP10 reproducible results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_run_dirs(root: str | Path) -> list[Path]:
    """Return run directories containing a run_manifest.json file."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path.parent for path in root_path.rglob("run_manifest.json"))


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize one structured experiment run directory."""
    path = Path(run_dir)
    run_manifest = _read_json(path / "run_manifest.json")
    test_metrics = _read_json(path / "test_metrics.json") if (path / "test_metrics.json").exists() else {}
    config = _read_json(path / "resolved_config.json") if (path / "resolved_config.json").exists() else {}
    env = _read_json(path / "environment_report.json") if (path / "environment_report.json").exists() else {}
    train = config.get("train", {}) if isinstance(config, dict) else {}
    return {
        "run_id": run_manifest.get("run_id", path.name),
        "run_dir": str(path),
        "dataset_task": run_manifest.get("dataset_task"),
        "dataset_seed": run_manifest.get("dataset_seed"),
        "dataset_manifest": run_manifest.get("dataset_manifest"),
        "checkpoint": run_manifest.get("checkpoint"),
        "model_name": train.get("model_name"),
        "device": train.get("device"),
        "seed": train.get("seed"),
        "batch_size": train.get("batch_size"),
        "num_epochs": train.get("num_epochs"),
        "lr": train.get("lr"),
        "python": env.get("python"),
        "torch": (env.get("packages") or {}).get("torch"),
        "cuda_available": (env.get("torch_runtime") or {}).get("cuda_available"),
        "split": test_metrics.get("split"),
        "count": test_metrics.get("count"),
        "mse": test_metrics.get("mse"),
        "rmse": test_metrics.get("rmse"),
        "mae": test_metrics.get("mae"),
        "normalized_mae_over_range": test_metrics.get("normalized_mae_over_range"),
        "p95_abs_error": test_metrics.get("p95_abs_error"),
        "max_abs_error": test_metrics.get("max_abs_error"),
    }


def collect_runs(root: str | Path) -> list[dict[str, Any]]:
    """Collect summaries for all run directories under root."""
    return [summarize_run(run_dir) for run_dir in discover_run_dirs(root)]


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def collect_and_write(root: str | Path, output_dir: str | Path) -> dict[str, str | int]:
    """Collect run summaries and write JSON and CSV benchmark tables."""
    rows = collect_runs(root)
    out = Path(output_dir)
    json_path = write_json(out / "benchmark_results.json", rows)
    csv_path = write_csv(out / "benchmark_results.csv", rows)
    return {"runs": len(rows), "json": str(json_path), "csv": str(csv_path)}
