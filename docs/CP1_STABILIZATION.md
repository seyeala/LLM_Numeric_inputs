# CP1 Stabilization

CP1 makes the repository runnable from a cleaner checkout for the validated first-stage workflows. It does not attempt to solve the full research problem or validate the mixed text-plus-numeric pathway.

## Scope

CP1 covers repository hygiene, configuration correctness, CLI consistency, checkpoint path handling, and import-time/runtime stability on CPU or CUDA hosts.

Validated target workflows:

1. Numeric-to-numeric projection alignment through `alignmentNN.py`.
2. Text-to-numeric projection alignment through `alignmentTextN.py`.

Explicitly out of scope:

1. Full benchmark reproduction.
2. Mixed text-plus-numeric alignment correctness.
3. Dependency modernization or lockfile generation.
4. CI creation.
5. A formal pytest suite.

## Implementation summary

CP1 implemented the following changes:

1. Added `.gitignore` and removed the committed `.DS_Store` file.
2. Rewrote `config/config.yaml` with unique, valid YAML keys.
3. Fixed the malformed `model_path_save::` key.
4. Added explicit defaults for device selection, model loading, and model saving.
5. Removed the duplicate `load_specific_weights` implementation from `wrapperNM.py`.
6. Made CUDA memory helper functions safe on CPU-only machines.
7. Added `map_location` when loading checkpoints.
8. Added checkpoint parent-directory creation before `torch.save`.
9. Fixed `alignmentNN.py` so load and save paths are separate.
10. Added CLI overrides for batch count, batch size, learning rate, model paths, and device.

## Files changed

Primary implementation files:

- `.gitignore`
- `config/config.yaml`
- `wrapperNM.py`
- `alignmentNN.py`
- `alignmentTextN.py`
- `alignmentMixedN.py`

Documentation files:

- `README.md`
- `docs/CP1_STABILIZATION.md`

`alignmentMixedN.py` received the same CP1-level operational cleanup as the other training scripts, but its model-path and CLI stabilization should not be interpreted as full mixed-mode validation.

## Verification commands

Run the syntax-level validation from the repository root:

```powershell
cd F:\LLM_Numeric_inputs
python -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py dataset\text_numeric_generator_for_llm.py
```

Expected result:

```text
py_compile: ok
```

A runtime smoke test requires `torch`, `transformers`, and `PyYAML` to be installed in the active Python environment.

## Runtime smoke-test commands

After installing the runtime dependencies, run:

```powershell
cd F:\LLM_Numeric_inputs
python alignmentNN.py --config .\config\config.yaml --num_epochs 1 --num_batches 1 --batch_size 1 --model_path_load "" --model_path_save .\chk\smoke_nn.pth
python alignmentTextN.py --config .\config\config.yaml --num_epochs 1 --num_batches 1 --batch_size 1 --model_path_load .\chk\smoke_nn.pth --model_path_save .\chk\smoke_textn.pth
```

These commands should create `chk\smoke_nn.pth` and `chk\smoke_textn.pth` without manually creating the `chk` directory.

## Current workstation limitation

On the tested workstation, the default interpreter is Python 3.14 and PyTorch is not installed in that environment. Importing `torch` fails with:

```text
ModuleNotFoundError: No module named 'torch'
```

Because of that environment limitation, CP1 was validated with `py_compile`, but not with an actual training run on the workstation.

The pinned dependency `torch==1.13.1` in the existing `requirements.txt` is also not appropriate for Python 3.14. Use a compatible Python environment or modernize the dependency set in a later checkpoint.

## Known limitations after CP1

1. No pytest suite has been added yet.
2. No GitHub Actions workflow has been added yet.
3. `requirements.txt` is still a broad environment dump and should be reduced in a later checkpoint.
4. `alignmentMixedN.py` still requires architectural review before it should be treated as a validated training path.
5. Benchmarks are not yet reproducible from a single command.
6. The README still contains historical claims and observations that should be separated from validated current results in a later documentation pass.

## Recommended next checkpoint

CP2 should add a minimal dependency file, a tiny-model CPU test path, pytest smoke tests, and CI. That will convert CP1 from syntax-level stabilization into automated runtime validation.
