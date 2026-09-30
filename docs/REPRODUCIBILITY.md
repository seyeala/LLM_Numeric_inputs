# Reproducibility guide

This repository is organized around checkpointed, source-backed workflows.
The supported local baseline is Python 3.12 with dependencies defined in `pyproject.toml` and `requirements*.txt`.

## Environment

Create the environment on Windows with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

On the RTX 3090 workstation, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Gpu
```

Record environment provenance with:

```powershell
.\.venv\Scripts\python.exe scripts\environment_report.py
```

## Data

New data should be generated through the CP5 deterministic pipeline.
Legacy JSONL files in `dataset/` are historical artifacts.

Example:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task numeric_to_numeric --count 1000 --seed 42 --output-dir dataset\generated\example_numeric
.\.venv\Scripts\python.exe scripts\validate_dataset.py --manifest dataset\generated\example_numeric\manifest.json
```

## Experiments

Use the checkpoint-specific training and evaluation scripts under `scripts/`.
Every structured run writes a run directory with config, environment metadata, metrics, checkpoint, and run manifest.

## Benchmark summaries

Aggregate local run directories with:

```powershell
.\.venv\Scripts\python.exe scripts\collect_benchmark_results.py --runs-root runs --output-dir benchmarks\generated
```

Generated benchmark outputs are local artifacts and are not committed.
