# CP2 — Reproducible environment and dependency baseline

Date: 2026-09-29
Repository: `F:\\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `d69c38e`

## Scope

CP2 establishes a supported Python runtime and replaces the original captured-environment install path with a minimal dependency baseline.

Supported workstation baseline:

- Python 3.12.10
- PyTorch 2.14.0+cu126
- CUDA build 12.6
- Transformers 5.17.0
- NumPy 2.5.3
- PyYAML 6.0.3
- pytest 9.1.1
- Ruff 0.16.9

The local virtual environment lives at `.venv` and is ignored by Git.
## Files changed

- `.gitignore`: ignores `.venv`, Ruff cache, build outputs, and package metadata.
- `pyproject.toml`: adds project metadata, Python constraint, direct runtime dependencies, and dev extras.
- `requirements.txt`: minimal runtime dependency list.
- `requirements-dev.txt`: runtime dependencies plus pytest and Ruff.
- `requirements-gpu-cu126.txt`: CUDA 12.6 PyTorch override for the RTX 3090 workstation.
- `requirements-lock-cu126.txt`: exact package freeze from the tested GPU environment.
- `requirements-legacy.txt`: preserved original broad environment dump for provenance only.
- `scripts/setup_env.ps1`: creates/updates the local Python 3.12 environment.
- `scripts/environment_report.py`: emits machine-readable runtime provenance.
- `docs/checkpoints/CP02_environment_report.json`: captured workstation environment report.

The legacy requirements file is retained only for historical reference and should not be used for new installs.
## Setup commands

Create or refresh the development environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

On the RTX 3090 workstation, verify or apply the CUDA 12.6 PyTorch build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Gpu
```

Inspect the active environment:

```powershell
.\.venv\Scripts\python.exe scripts\environment_report.py
```
## Validation commands executed

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Gpu
.\.venv\Scripts\python.exe -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py dataset\text_numeric_generator_for_llm.py scripts\environment_report.py
.\.venv\Scripts\python.exe -c "import torch, transformers, yaml, numpy; print(torch.__version__, transformers.__version__, yaml.__version__, numpy.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check scripts\environment_report.py
.\.venv\Scripts\python.exe scripts\environment_report.py --output docs\checkpoints\CP02_environment_report.json
```

## Validation results

- Editable project install succeeded through `pyproject.toml`.
- `pip check` returned `No broken requirements found.`
- `py_compile` returned `py_compile: ok`.
- Runtime imports succeeded.
- PyTorch reported `2.14.0+cu126`.
- CUDA is available and PyTorch sees 2 devices.
- `nvidia-smi` and PyTorch both identify two NVIDIA GeForce RTX 3090 GPUs with 24,576 MiB each.
- Ruff passed for `scripts/environment_report.py`.
- `docs/checkpoints/CP02_environment_report.json` records the package versions, CUDA state, GPU identities, Git SHA, and working-tree state at capture time.

## Acceptance checklist

- [x] Python 3.12 virtual environment created at `.venv`.
- [x] Documented setup command succeeds.
- [x] `torch`, `transformers`, `yaml`, and `numpy` import successfully.
- [x] PyTorch reports the two RTX 3090 GPUs in the GPU environment.
- [x] `py_compile` passes for project scripts and the environment report script.
- [x] Active direct dependency file no longer contains the old broad environment dump or the invalid `data` entry.
- [x] Exact tested GPU environment is frozen in `requirements-lock-cu126.txt`.

## Remaining limitations

CP2 does not validate model training. Runtime training validation begins in later checkpoints after core contracts and tests are added.
