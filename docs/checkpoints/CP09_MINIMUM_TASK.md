# CP9 - Minimum-number mixed-task experiment

Date: 2026-09-29
Repository: `F:\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `5b08ab9`

## Scope

CP9 validates the repaired CP8 mixed text/numeric architecture on the repository's minimum-number task.
It establishes the mixed-task experiment pipeline, not a final GPT-2 benchmark claim.

## Implementation

Added minimum-task experiment modules:

```text
src/llm_numeric_inputs/experiments/minimum_data.py
src/llm_numeric_inputs/experiments/minimum_evaluation.py
src/llm_numeric_inputs/experiments/minimum_alignment.py
```

Added CLI scripts:

```text
scripts/train_minimum_mixed.py
scripts/evaluate_minimum_mixed.py
```

Added configs:

```text
configs/cp9_minimum_smoke.yaml
configs/cp9_minimum_tiny_cuda.yaml
configs/cp9_minimum_gpt2.yaml
```

Added tests:

```text
tests/test_minimum_mixed.py
```

## Model contract

The CP9 model uses:

- `InputMode.MIXED_TO_NUMERIC`,
- projected numeric embeddings inserted at `$$value&&` marker positions,
- ordinary token embeddings for text segments,
- frozen base transformer by default,
- trainable input and output projection layers.

## Smoke runs completed

CPU smoke dataset and run:

```text
dataset/generated/cp9_minimum_smoke
runs/cp9_minimum_smoke/cp9_minimum_smoke
```

CUDA tiny-model smoke dataset and run:

```text
dataset/generated/cp9_minimum_tiny_cuda
runs/cp9_minimum_tiny_cuda/cp9_minimum_tiny_cuda
```

The CPU smoke evaluation produced test metrics on 6 held-out examples. These metrics verify pipeline execution only and are not reported as scientific benchmark results.

## Validation commands run

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\train_minimum_mixed.py scripts\evaluate_minimum_mixed.py src\llm_numeric_inputs\experiments\minimum_alignment.py src\llm_numeric_inputs\experiments\minimum_data.py src\llm_numeric_inputs\experiments\minimum_evaluation.py
.\.venv\Scripts\python.exe -m pytest tests\test_minimum_mixed.py tests\test_wrapper_mixed.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check src scripts tests alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py
```

## Known limitations after CP9

- GPT-2/GPT-2 Large minimum-task benchmarks are deferred to CP10.
- Template-holdout generalization is not yet implemented; CP9 uses deterministic CP5 split files.
- The smoke metrics are from tiny local models and should not be interpreted as model-quality evidence.

## Acceptance checklist

- [x] Mixed minimum-task dataset loading exists.
- [x] Mixed minimum-task evaluation exists.
- [x] Mixed minimum-task training exists.
- [x] CPU smoke run completes.
- [x] CUDA tiny-model smoke run completes.
- [x] Tests cover split loading, metrics, and training run output.
- [x] Projection-only schema-v1 checkpoints are used.
- [x] Documentation is written.

## Next checkpoint

CP10 should add reproducible benchmark aggregation, structured run collection, and legacy-result separation.
