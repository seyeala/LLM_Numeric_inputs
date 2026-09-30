# CP5 - Deterministic dataset pipeline

Date: 2026-09-29
Repository: `F:\\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `385027f`

## Scope

CP5 replaces ad hoc generated data workflows with deterministic generation, manifesting, and streaming validation.
It does not run model training or benchmark experiments.

## Data package

New modules under `src/llm_numeric_inputs/data/`:

- `schemas.py`: task names, split names, schema constants.
- `generators.py`: pure seeded record generators.
- `splits.py`: deterministic train/validation/test split counts and index assignment.
- `manifests.py`: SHA-256, row counts, generator metadata, manifest writing.
- `validation.py`: streaming JSONL and manifest validation.
- `dataset.py`: high-level split-file generation.

Supported tasks:

```text
numeric_to_numeric
text_to_numeric
minimum
```

## Schema

Every JSONL row uses the simple two-field schema:

```json
{"input": "...", "output": "..."}
```

Task-specific validation rules:

- `numeric_to_numeric`: numeric input equals numeric output.
- `text_to_numeric`: input is exactly `$$value&&` and output equals `value`.
- `minimum`: output equals the minimum numeric marker shown in the input text.

## Generation and validation commands

Generate deterministic data:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task minimum --count 50 --seed 42 --output-dir data\generated\cp5_smoke_minimum
```

Validate a manifest:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py --manifest data\generated\cp5_smoke_minimum\manifest.json
```

Each generated dataset directory contains:

```text
train.jsonl
validation.jsonl
test.jsonl
manifest.json
```

## Manifest contents

`manifest.json` records:

- schema version,
- task,
- seed,
- total count,
- split counts,
- numeric range,
- generator Git SHA,
- Python and NumPy versions,
- per-file row counts and SHA-256 hashes,
- generation parameters such as decimal precision and split ratios.

## Legacy data policy

The existing large files in `dataset/` remain tracked as legacy artifacts:

- `dataset/alignmentNN.jsonl`
- `dataset/alignmentTN.jsonl`
- `dataset/minimumalgo.jsonl`

They are retained for provenance only. New bulk output should be written under ignored generated-data directories:

```text
data/generated/
dataset/generated/
```

`dataset/README.md` documents the legacy status and the CP5 replacement workflow.

## Compatibility generator

`dataset/text_numeric_generator_for_llm.py` is now a no-side-effect compatibility wrapper.
Importing it no longer writes files.
The old helper function names remain, but new code should use `scripts/generate_dataset.py` or `llm_numeric_inputs.data.generate_dataset`.

## Fixture data

Small deterministic fixture datasets are tracked under:

```text
tests/fixtures/generated/
```

They cover all three supported tasks and include manifests for tests and examples.

## Tests added

CP5 adds tests for:

- no import-time file writes,
- same-seed deterministic generation,
- different-seed data changes,
- label correctness for all supported tasks,
- split/manifest generation,
- manifest SHA-256 checks,
- streaming validation failures for malformed records,
- generate/validate CLI execution.

## Validation commands run

```powershell
.\.venv\Scripts\python.exe -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py scripts\environment_report.py scripts\cp3_smoke_contracts.py scripts\generate_dataset.py scripts\validate_dataset.py dataset\text_numeric_generator_for_llm.py
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task minimum --count 50 --seed 42 --output-dir data\generated\cp5_smoke_minimum
.\.venv\Scripts\python.exe scripts\validate_dataset.py --manifest data\generated\cp5_smoke_minimum\manifest.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check src scripts tests alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py dataset\text_numeric_generator_for_llm.py
```

## Validation result

```text
Generated dataset manifest: data\generated\cp5_smoke_minimum\manifest.json
Dataset validation ok: task=minimum count=50 seed=42
71 passed
No broken requirements found.
All checks passed!
```

## Acceptance checklist

- [x] Importing generator modules writes no files.
- [x] Deterministic generation works for numeric_to_numeric, text_to_numeric, and minimum.
- [x] Same seed produces identical records/hashes.
- [x] Different seed changes generated data.
- [x] Train/validation/test splits are deterministic and documented.
- [x] Manifest includes schema version, task, seed, counts, file hashes, and generator metadata.
- [x] Streaming validator catches malformed rows and wrong labels.
- [x] Small fixture data is tracked for tests.
- [x] Bulk generated output paths are ignored.
- [x] Legacy JSONL files are documented as legacy artifacts.
- [x] pytest passes.
- [x] Ruff passes.
- [x] pip check passes.

## Known limitations after CP5

- The legacy large JSONL files are still tracked; history/LFS migration remains a separate repository-maintenance decision.
- CP5 validates data, not model-training quality.
- Template holdout logic for the minimum task is not implemented yet; CP9 will define task-specific train/test generalization splits.

## Next checkpoint

CP6 should use the deterministic data pipeline to run and evaluate numeric-to-numeric alignment with independent validation/test metrics.
