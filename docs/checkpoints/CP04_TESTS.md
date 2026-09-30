# CP4 - Automated offline pytest suite

Date: 2026-09-29
Repository: `F:\\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `14ba4ec`

## Scope

CP4 converts the CP3 smoke-contract checks into a formal offline pytest suite.
The tests validate configuration handling, device helpers, pooling behavior, wrapper contracts, trainability, checkpoint schema behavior, lightweight training utilities, and compatibility entry points.

CP4 does not validate scientific training performance. That remains CP6 and later.

## Offline test design

The test suite uses tiny local dummy objects instead of Hugging Face downloads:

- `DummyCausalLM`: a small local causal-LM-like module with hidden states.
- `PositionCausalLM`: a deterministic hidden-state generator used to verify padding-aware pooling.
- `DummyTokenizer`: a minimal tokenizer-like object that returns tensor dictionaries with a `.to()` method.

The fixtures set:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

No CP4 test requires network access or an external model repository.

## Test inventory

Files added under `tests/`:

- `conftest.py`: offline fixtures and dummy model/tokenizer classes.
- `test_config.py`: config defaults, YAML loading, CLI overrides, validation, path normalization.
- `test_devices.py`: device resolution and CPU-safe CUDA utility behavior.
- `test_pooling.py`: final-valid-token pooling and invalid-mask/shape failures.
- `test_wrapper_numeric.py`: numeric wrapper shape, invalid inputs, numeric sequence strategies, projection gradients.
- `test_wrapper_text.py`: text wrapper shape, required tokenized inputs, missing masks, padding-aware pooling.
- `test_trainability.py`: trainability flags and frozen-transformer invariance after an optimizer step.
- `test_checkpointing.py`: schema-v1 checkpoint fields, projection-only state, load round trip, legacy state_dict loading, invalid payloads.
- `test_training_utils.py`: generated batch shapes, one-step training loop, no-trainable-parameter failure.
- `test_entrypoints.py`: root entrypoint imports/parsers and mixed-mode CP8 deferral message.

## Implementation notes

CP4 made one small runtime hardening change in `training_utils.py`: the shared training loop now raises a clear `ValueError` when no parameters are trainable instead of delegating to an optimizer error.

CP4 also modernized AMP imports in `training_utils.py` from `torch.cuda.amp` to `torch.amp`, removing deprecation warnings on the CP2 PyTorch environment.

Documentation mojibake found in roadmap/checkpoint docs was normalized to ASCII punctuation.

## Validation commands run

```powershell
.\.venv\Scripts\python.exe -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py scripts\environment_report.py scripts\cp3_smoke_contracts.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check src scripts tests alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py
```

## Validation result

```text
py_compile: ok
56 passed
No broken requirements found.
All checks passed!
```

## Acceptance checklist

- [x] `tests/` exists.
- [x] `pytest -q` runs offline.
- [x] Tests do not download Hugging Face models/tokenizers.
- [x] Numeric wrapper shape and invalid-input tests pass.
- [x] Text wrapper shape and padding-pooling tests pass.
- [x] Trainability/freeze tests pass.
- [x] Checkpoint schema and legacy loading tests pass.
- [x] One-step training utility smoke test passes.
- [x] Entrypoint import/parser tests pass.
- [x] Documentation encoding cleanup is complete.
- [x] Ruff passes.
- [x] pip check passes.
- [x] py_compile passes.

## Known limitations after CP4

- Dataset generation is still not deterministic or side-effect-free; this is CP5.
- Numeric/text experiments are not scientifically validated yet; these are CP6 and CP7.
- Mixed input remains intentionally disabled until CP8.
- No CI workflow exists yet; CI is scheduled for CP11.

## Next checkpoint

CP5 should refactor the dataset generator into a deterministic, side-effect-free CLI with manifest and split generation.
