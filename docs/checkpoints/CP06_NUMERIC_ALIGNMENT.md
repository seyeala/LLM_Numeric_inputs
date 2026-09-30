# CP6 - Numeric-to-numeric validated experiment pipeline

Date: 2026-09-29
Repository: `F:\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `185c4bb`

## Scope

CP6 adds the first dataset-backed numeric-to-numeric training and evaluation pipeline. It uses CP5 `numeric_to_numeric` manifests, writes structured run directories, saves projection-only checkpoints, and reports independent validation/test metrics.

CP6 validates the experiment machinery with offline tiny-model runs and one CUDA tiny-model run. It does not claim a reproduced GPT-2 or GPT-2 Large research result; that belongs to CP10 once benchmark protocol and runtime budgets are fixed.

## New package modules

Added under `src/llm_numeric_inputs/experiments/`:

- `metrics.py`: MSE, RMSE, MAE, normalized MAE, relative error, quantiles, max error.
- `numeric_data.py`: CP5 manifest loading, numeric split parsing, deterministic batching.
- `evaluation.py`: batched numeric prediction and split evaluation.
- `run_metadata.py`: run IDs, Git/environment/package metadata, JSON artifact writing.
- `tiny_model.py`: tiny local causal-LM-like module for offline smoke tests.
- `numeric_alignment.py`: config loading, model construction, training loop, checkpointing, evaluation.

## New scripts and configs

Scripts:

```text
scripts/train_numeric_alignment.py
scripts/evaluate_numeric_alignment.py
```

Configs:

```text
configs/cp6_numeric_smoke.yaml
configs/cp6_numeric_tiny_cuda.yaml
configs/cp6_numeric_gpt2.yaml
```

The smoke and tiny-CUDA configs use `__tiny_numeric_lm__` and do not require Hugging Face downloads.

## Run artifact layout

Each CP6 training run writes a run directory containing:

```text
checkpoint.pth
resolved_config.json
environment_report.json
epoch_metrics.json
test_metrics.json
run_manifest.json
```

The checkpoint is schema-v1 projection-only and remains compatible with the CP3 checkpoint loader.

## Metrics

Each evaluation split reports:

```text
count
mse
rmse
mae
mean_absolute_relative_error
normalized_mae_over_range
p50_abs_error
p90_abs_error
p95_abs_error
max_abs_error
split
task
dataset_count
```

`normalized_mae_over_range` is defined as `mae / (max_val - min_val)`.

## Smoke commands run

Generate smoke data:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task numeric_to_numeric --count 64 --seed 606 --output-dir dataset\generated\cp6_numeric_smoke
```

Train offline tiny model:

```powershell
.\.venv\Scripts\python.exe scripts\train_numeric_alignment.py --config configs\cp6_numeric_smoke.yaml --manifest dataset\generated\cp6_numeric_smoke\manifest.json --output-dir runs\cp6_numeric_smoke
```

Evaluate the produced checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_numeric_alignment.py --checkpoint runs\cp6_numeric_smoke\cp6_numeric_smoke\checkpoint.pth --manifest dataset\generated\cp6_numeric_smoke\manifest.json --split test --output runs\cp6_numeric_smoke\cp6_numeric_smoke\test_eval_repeat.json
```

CUDA tiny-model smoke run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task numeric_to_numeric --count 32 --seed 607 --output-dir dataset\generated\cp6_numeric_tiny_cuda
.\.venv\Scripts\python.exe scripts\train_numeric_alignment.py --config configs\cp6_numeric_tiny_cuda.yaml --manifest dataset\generated\cp6_numeric_tiny_cuda\manifest.json --output-dir runs\cp6_numeric_tiny_cuda
```

## Smoke result summary

The offline smoke run completed and wrote:

```text
runs\cp6_numeric_smoke\cp6_numeric_smoke\checkpoint.pth
runs\cp6_numeric_smoke\cp6_numeric_smoke\test_metrics.json
runs\cp6_numeric_smoke\cp6_numeric_smoke\test_eval_repeat.json
```

The repeated test evaluation returned a test split with 7 examples. Metrics are intentionally poor because the tiny model is only a pipeline smoke fixture and was trained for two epochs on 64 generated examples.

The CUDA smoke run completed on `cuda:0` with the CP2 CUDA 12.6 PyTorch environment and detected two RTX 3090 GPUs.

## Tests added

CP6 added tests for:

- regression metric correctness and invalid inputs,
- numeric split loading from CP5 manifests,
- numeric evaluation metrics,
- end-to-end tiny-model training run artifact creation,
- schema-v1 checkpoint metadata from the CP6 training path.

## Validation commands run

```powershell
.\.venv\Scripts\python.exe -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py scripts\environment_report.py scripts\cp3_smoke_contracts.py scripts\generate_dataset.py scripts\validate_dataset.py scripts\train_numeric_alignment.py scripts\evaluate_numeric_alignment.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check src scripts tests alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py dataset\text_numeric_generator_for_llm.py
```

## Validation result

```text
py_compile: ok
77 passed
No broken requirements found.
All checks passed!
```

## Acceptance checklist

- [x] Dataset-backed numeric_to_numeric trainer exists.
- [x] Independent validation and test evaluation exists.
- [x] Metrics include MAE, RMSE, MSE, normalized MAE, quantiles, and count.
- [x] Run directory includes config, environment, dataset manifest reference, metrics, and checkpoint.
- [x] Checkpoint remains projection-only schema-v1 compatible.
- [x] Smoke training/evaluation runs offline with a tiny local model.
- [x] A CUDA smoke run completes on the workstation using a tiny local model.
- [x] Historical numeric-alignment claims are not repeated.
- [x] pytest passes.
- [x] Ruff passes.
- [x] pip check passes.

## Known limitations after CP6

- CP6 does not reproduce historical GPT-2/GPT-2 Large numeric alignment results.
- The tiny local model smoke metrics are not scientifically meaningful.
- The default training loop uses deterministic ordered batches; shuffled epoch sampling can be added in a later experiment checkpoint if needed.
- Canonical GPT-2 benchmark runs are deferred to CP10.

## Next checkpoint

CP7 should extend the same dataset-backed experiment framework to text-to-numeric alignment and compare fresh output-projection initialization with transfer from CP6 numeric alignment.
