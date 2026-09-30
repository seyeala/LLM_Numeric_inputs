# CP3 - Core contracts and checkpoint schema

Date: 2026-09-29
Repository: `F:\\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `41d6511`

## Scope

CP3 converts the prototype scripts into a small installable package with explicit internal contracts.
It does not attempt to produce training-quality scientific results; that remains CP6 and later.

## Package structure

New package root:

```text
src/llm_numeric_inputs/
```

Modules added:

- `config.py`: `TrainConfig`, YAML loading, CLI overrides, validation.
- `devices.py`: device resolution plus CUDA utility functions.
- `seeding.py`: shared Python/NumPy/Torch seeding.
- `modes.py`: explicit `InputMode` and `NumericSequenceStrategy` enums.
- `pooling.py`: attention-mask-aware last-valid-token pooling.
- `wrapper.py`: CP3 `NumericLMWrapper` contracts.
- `checkpointing.py`: schema-v1 projection checkpoint save/load and legacy compatibility.
- `training_utils.py`: shared lightweight training-loop utilities for legacy entry scripts.

`pyproject.toml` now uses `src` package discovery instead of installing only a root module.

## Formal wrapper contracts

CP3 defines these supported contracts:

```text
numeric_to_numeric: Tensor[batch, 1] -> Tensor[batch, 1]
text_to_numeric: tokenized batch dict -> Tensor[batch, 1]
mixed_to_numeric: intentionally disabled until CP8
```

The mixed path now fails deliberately with a clear message rather than running a known-invalid architecture.

## Numeric sequence strategy

Default numeric input handling changed from repeating one projected scalar across the model's full context length to a single projected sequence token.

Default:

```text
numeric_sequence_strategy = "single_token"
```

The legacy behavior remains explicit as:

```text
numeric_sequence_strategy = "repeat_n_positions"
```

## Text pooling rule

Text-to-numeric output now pools the hidden state at the final valid token according to `attention_mask`.
It no longer assumes `last_hidden_state[:, -1, :]`, which can select padding.

## Checkpoint schema

CP3 introduces projection-only schema-v1 checkpoints. Default checkpoints contain:

- `schema_version`,
- task mode,
- model name,
- numeric sequence strategy,
- training config,
- projection state dict,
- metadata including timestamp, Git SHA, Python, Torch, and Transformers versions.

The checkpoint loader also supports legacy full `state_dict` files for compatibility.

## Compatibility entry points

The root scripts remain available:

- `alignmentNN.py`
- `alignmentTextN.py`
- `alignmentMixedN.py`

`alignmentNN.py` and `alignmentTextN.py` now call the shared package modules.
`alignmentMixedN.py` is disabled until CP8 because the previous mixed architecture was invalid.

`wrapperNM.py` remains as a compatibility shim and re-exports the package wrapper/utilities.

## CP3 smoke validation

Smoke script:

```powershell
.\.venv\Scripts\python.exe scripts\cp3_smoke_contracts.py
```

It validates:

- numeric wrapper output shape,
- text wrapper output shape,
- attention-aware pooling,
- projection-only checkpoint save/load,
- legacy full-state checkpoint loading.

## Validation commands run

```powershell
.\.venv\Scripts\python.exe -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py scripts\environment_report.py scripts\cp3_smoke_contracts.py
.\.venv\Scripts\python.exe scripts\cp3_smoke_contracts.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check src scripts alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py
```

## Validation result

```text
py_compile: ok
CP3 smoke contracts: ok
No broken requirements found.
All checks passed!
```

## Acceptance checklist

- [x] `src/llm_numeric_inputs` package exists.
- [x] `pyproject.toml` uses package discovery.
- [x] Config loading and override logic are centralized.
- [x] Device resolution is centralized and validated.
- [x] Seeding is centralized.
- [x] Task modes are explicit.
- [x] Numeric forward has a documented `[B,1] -> [B,1]` contract.
- [x] Text forward has a documented tokenized-batch -> `[B,1]` contract.
- [x] Text pooling uses `attention_mask`.
- [x] Numeric mode defaults to one projected sequence token.
- [x] Legacy repeat-context behavior is explicit.
- [x] Checkpoint schema v1 saves projection state plus metadata.
- [x] Legacy full-state checkpoint loading is supported.
- [x] Old entry scripts remain present.
- [x] Mixed mode is clearly deferred to CP8.
- [x] CP3 validation commands pass.

## Known limitations after CP3

- No formal pytest suite exists yet; this is CP4.
- The dataset generator is still not deterministic/side-effect-free; this is CP5.
- Numeric/text experiments are not scientifically validated yet; these are CP6 and CP7.
- Mixed input is not implemented; it is deliberately deferred to CP8.
- The old root scripts remain compatibility wrappers and may be removed or replaced by console commands in CP11.

## Next checkpoint

CP4 should convert the smoke checks into an offline pytest suite with tiny local model fixtures.
