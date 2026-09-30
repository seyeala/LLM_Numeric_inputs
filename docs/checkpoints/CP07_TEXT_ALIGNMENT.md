# CP7 - Text-to-numeric validation

Date: 2026-09-29
Repository: `F:\\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `592ea3a`

## Scope

CP7 adds dataset-backed text-to-numeric training and evaluation using CP5 `text_to_numeric` manifests.
It validates the text experiment pipeline, not the final GPT-2 benchmark claim.

## Implementation summary

New experiment modules:

- `src/llm_numeric_inputs/experiments/text_data.py`: text split loading, deterministic batching, tokenization helper.
- `src/llm_numeric_inputs/experiments/text_evaluation.py`: batched text prediction and text-to-numeric metrics.
- `src/llm_numeric_inputs/experiments/text_alignment.py`: CP7 training loop, config loading, model construction, checkpointing.

New entry scripts:

- `scripts/train_text_alignment.py`
- `scripts/evaluate_text_alignment.py`

New configs:

- `configs/cp7_text_smoke.yaml`
- `configs/cp7_text_tiny_cuda.yaml`
- `configs/cp7_text_gpt2.yaml`

New tests:

- `tests/test_text_data.py`
- `tests/test_text_evaluation.py`
- `tests/test_text_alignment.py`

## Experiment contract

Input data must come from a CP5 manifest with:

```text
task = text_to_numeric
```

The model contract is:

```text
input: text strings such as $$12.3400&&
tokenization: tokenizer-dependent tokenized batch dict
output: Tensor[batch, 1]
```

The default trainability policy is:

```text
input_projection: frozen / unused
output_projection: trainable
transformer: frozen
```

## Initialization modes

CP7 supports two initialization modes:

```text
fresh
transfer
```

`fresh` initializes the output projection from the model constructor.

`transfer` loads compatible projection weights from a CP3+ schema checkpoint or legacy compatible checkpoint. This enables comparison against CP6 numeric-to-numeric projection initialization without assuming it helps.

Transfer support is validated by tests and by a local smoke run using a CP6 smoke checkpoint.

## Metrics

CP7 reports the same regression metrics as CP6:

- MSE,
- RMSE,
- MAE,
- mean absolute relative error,
- normalized MAE over configured range,
- p50/p90/p95 absolute error,
- max absolute error,
- count.

Validation metrics are recorded per epoch. Test metrics are recorded once after training.

## Smoke commands run

Generate a deterministic CP7 smoke dataset:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task text_to_numeric --count 64 --seed 707 --output-dir dataset\generated\cp7_text_smoke
```

Train the CPU tiny-model smoke run:

```powershell
.\.venv\Scripts\python.exe scripts\train_text_alignment.py --config configs\cp7_text_smoke.yaml --manifest dataset\generated\cp7_text_smoke\manifest.json --output-dir runs\cp7_text_smoke
```

Evaluate the CPU tiny-model smoke checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_text_alignment.py --checkpoint runs\cp7_text_smoke\cp7_text_smoke\checkpoint.pth --manifest dataset\generated\cp7_text_smoke\manifest.json --split test --batch-size 8 --device cpu
```

Train the CUDA tiny-model smoke run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task text_to_numeric --count 64 --seed 708 --output-dir dataset\generated\cp7_text_tiny_cuda
.\.venv\Scripts\python.exe scripts\train_text_alignment.py --config configs\cp7_text_tiny_cuda.yaml --manifest dataset\generated\cp7_text_tiny_cuda\manifest.json --output-dir runs\cp7_text_tiny_cuda
```

A local transfer-initialization smoke run was also completed using the CP6 smoke checkpoint as the transfer source. The ad hoc transfer config was not committed because it points to local generated run artifacts.

## Smoke results

CPU tiny-model test metrics from the CP7 smoke run:

```text
count: 7
mse: 3083.192138671875
rmse: 55.526499427497455
mae: 46.43299102783203
normalized_mae_over_range: 0.4643299102783203
split: test
task: text_to_numeric
```

These values are pipeline smoke metrics, not scientific claims.

## Validation commands run

```powershell
.\.venv\Scripts\python.exe -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py scripts\environment_report.py scripts\cp3_smoke_contracts.py scripts\generate_dataset.py scripts\validate_dataset.py scripts\train_numeric_alignment.py scripts\evaluate_numeric_alignment.py scripts\train_text_alignment.py scripts\evaluate_text_alignment.py dataset\text_numeric_generator_for_llm.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check src scripts tests alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py dataset\text_numeric_generator_for_llm.py
```

## Validation result

```text
py_compile: ok
84 passed
No broken requirements found.
All checks passed!
```

## Acceptance checklist

- [x] Dataset-backed text_to_numeric trainer exists.
- [x] Dataset-backed text_to_numeric evaluator exists.
- [x] Fresh initialization is supported.
- [x] Transfer initialization is supported.
- [x] Validation/test split metrics are independent of training batches.
- [x] Output uses schema-v1 projection checkpointing.
- [x] Offline tiny-model tests pass.
- [x] CPU smoke training/evaluation runs.
- [x] CUDA tiny-model smoke training runs on the workstation.
- [x] pytest passes.
- [x] Ruff passes.
- [x] pip check passes.

## Known limitations after CP7

- CP7 does not claim GPT-2 text-to-numeric benchmark performance.
- CP7 does not yet run a full fresh-vs-transfer replicated study; CP10 will own canonical benchmark tables.
- CP7 uses simple CP5 text-to-numeric data, not robustness/OOD formatting sets. Robustness sets should be added before CP10 benchmarks.
- Mixed text-plus-numeric input remains disabled until CP8.

## Next checkpoint

CP8 should replace the disabled mixed path with a correct sequence-embedding architecture that inserts projected numeric values at marker positions without changing transformer hidden width.
