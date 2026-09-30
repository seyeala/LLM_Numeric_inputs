# Completion Roadmap - CP2 through CP11

This roadmap defines the ten remaining checkpoints required to turn `LLM_Numeric_inputs` from an experimental prototype into a reproducible, tested research repository.

CP1 is complete as a syntax/configuration stabilization checkpoint.
The next ten checkpoints are CP2, CP3, CP4, CP5, CP6, CP7, CP8, CP9, CP10, and CP11.

## Definition of finished

The project is considered complete only when all of the following are true:

- a fresh clone installs in a documented Python environment,
- unit and smoke tests pass without network access,
- numeric-to-numeric, text-to-numeric, and mixed text/numeric paths have explicit tensor contracts,
- all three paths have deterministic training and independent evaluation,
- dataset generation is seeded, versioned, and side-effect free,
- checkpoints are small, versioned, and contain enough metadata to reproduce a run,
- the mixed-input architecture correctly places numeric embeddings into the token sequence,
- the minimum-number task has reproducible baselines and results,
- benchmark runs record code, model, data, environment, timing, memory, and metrics,
- CI passes from a clean checkout,
- README claims match reproducible artifacts rather than historical notes.

Performance success and engineering completion are separate: a checkpoint may complete with a negative scientific result if the experiment is valid and fully documented.

## Checkpoint dependency chain

```text
CP2 Environment
  -> CP3 Core contracts
    -> CP4 Automated tests
      -> CP5 Deterministic data
        -> CP6 Numeric-to-numeric validation
          -> CP7 Text-to-numeric validation
            -> CP8 Mixed architecture
              -> CP9 Minimum-task experiment
                -> CP10 Reproducible benchmark suite
                  -> CP11 Release hardening
```

Each checkpoint should land as a separate commit or small commit series and must include a closure note under `docs/checkpoints/`.

---

## CP2 - Reproducible environment and dependency baseline


## Objective

Create one supported local environment that can actually import and run the project on this workstation.


## Deliverables

- Standardize the development/runtime baseline on Python 3.12.
- Create a local `.venv` using the available Python 3.12 interpreter.
- Replace the 224-line environment dump with a minimal direct dependency specification.
- Add `pyproject.toml` with runtime and development dependencies.
- Resolve and record tested package versions after installation.
- Keep CPU execution supported even on a CUDA workstation.
- Add an environment-report command or script that records Python, PyTorch, Transformers, CUDA availability, and GPU names.
- Document GPU installation separately from generic Python dependencies so the package is not tied to one CUDA wheel index.

### Acceptance criteria

- `py -3.12 -m venv .venv` succeeds.
- The documented install command succeeds from a clean environment.
- `python -c "import torch, transformers, yaml"` succeeds.
- PyTorch reports the two RTX 3090 GPUs when the GPU environment is used.
- `python -m py_compile` still passes for project Python files.
- No dependency named `data` or unrelated CUDA 11/CUDA 12 package mixture remains in the direct dependency list.
- Exact tested versions are written to an environment/lock artifact.


## CP2 closure evidence

Record the successful install commands, package versions, CUDA status, and the environment-report output in `docs/checkpoints/CP02_ENVIRONMENT.md`.

---

## CP3 - Core library, configuration, and checkpoint contracts


## Objective

Turn the current scripts into explicit reusable components before adding more experiments.


## Deliverables

- Create an installable `src/llm_numeric_inputs/` package.
- Centralize configuration loading, validation, CLI overrides, device selection, and seed handling.
- Define explicit modes: numeric-to-numeric, text-to-numeric, and mixed-to-numeric.
- Make the base model/tokenizer injectable so tests can use tiny local components.
- Define tensor contracts and validate input/output shapes.
- Replace padded-position output selection with attention-aware last-valid-token pooling.
- Define numeric input as one projected sequence token by default; keep the legacy full-sequence repeat only as an explicit ablation option.
- Introduce a versioned checkpoint schema with configuration, model identifier/revision, task mode, seed, and projection weights.
- Save projection weights only by default.
- Keep a compatibility loader for legacy full `state_dict` checkpoints.

### Acceptance criteria

- One shared configuration implementation is used by all CLIs.
- Numeric forward output is always `[batch, 1]`.
- Text forward output is always `[batch, 1]`.
- Padding does not change which valid token is pooled.
- A projection-only GPT-2 checkpoint is orders of magnitude smaller than a full GPT-2 state dict.
- A saved checkpoint round-trips into an equivalent model configuration.
- Legacy checkpoints either load through a documented compatibility path or fail with a precise migration message.


## CP3 closure evidence

Document the public API, tensor contracts, checkpoint format version, and legacy compatibility in `docs/checkpoints/CP03_CORE_CONTRACTS.md`.

---

## CP4 - Automated offline test harness


## Objective

Make regressions detectable before running expensive training.


## Deliverables

- Add `tests/` using pytest.
- Build tiny local model fixtures that do not download Hugging Face weights.
- Test configuration parsing and CLI overrides.
- Test CPU device resolution and checkpoint directory creation.
- Test numeric forward shape and gradients.
- Test text forward shape with variable-length padded examples.
- Test freeze/unfreeze behavior and confirm frozen transformer weights do not update.
- Test checkpoint save/load and legacy-loader behavior.
- Add one optimizer-step smoke test for numeric and text modes.
- Add basic static quality tooling such as Ruff or equivalent.

### Acceptance criteria

- `pytest -q` passes offline on CPU.
- Tests do not require a Hugging Face network download.
- At least one test would fail if text pooling used the padded final position.
- At least one test verifies transformer parameters remain bitwise unchanged in a projection-only training step.
- Numeric and text smoke tests complete with finite loss.
- Static checks pass using the documented command.

Mixed-mode functional tests are intentionally deferred until CP8; CP4 may assert that unsupported mixed behavior fails explicitly rather than silently.


## CP4 closure evidence

Store test inventory, commands, and results in `docs/checkpoints/CP04_TESTS.md`.

---

## CP5 - Deterministic dataset pipeline


## Objective

Replace ad hoc generated files with a reproducible data-generation contract.


## Deliverables

- Refactor `text_numeric_generator_for_llm.py` into pure functions plus a guarded CLI.
- Remove all dataset generation at import time.
- Seed Python and NumPy RNGs from one explicit seed.
- Add task-specific generation commands for N->N, text->N, and minimum-number data.
- Add explicit output directory handling.
- Generate deterministic train/validation/test splits.
- Add a manifest containing schema version, seed, counts, ranges, generator commit, and SHA-256 hashes.
- Add streaming validation for JSONL schema and label correctness.
- Track only small fixtures in Git by default; treat million-row files as generated artifacts.
- Preserve the current 2.1M-row files as legacy data until migration is verified.

### Acceptance criteria

- Importing the generator writes no files.
- Re-running the same command with the same seed produces identical hashes.
- Changing the seed changes the generated data.
- Every generated split passes schema and label validation.
- Train/validation/test files are disjoint according to the documented split rule.
- Small fixture data is sufficient for offline tests.
- The repository no longer requires ~105 MB of generated JSONL files for a normal test checkout.

History rewriting or Git LFS migration is optional and must be treated as a separate destructive repository-maintenance decision.


## CP5 closure evidence

Write generator commands, manifest examples, data schema, and migration notes in `docs/checkpoints/CP05_DATA.md`.

---

## CP6 - Numeric-to-numeric validated experiment


## Objective

Establish the first end-to-end scientifically reproducible training path.


## Deliverables

- Create a shared trainer/evaluator instead of training only inside `alignmentNN.py`.
- Use deterministic seeds and independent validation/test data.
- Record train loss and evaluation metrics separately.
- Report MAE, RMSE, normalized error over the configured numeric range, and error quantiles.
- Record config, Git SHA, checkpoint ID, package versions, model revision, device, runtime, and peak VRAM.
- Run the default one-numeric-token architecture.
- Keep the legacy 1,024-repeat behavior as an ablation if it remains scientifically useful.
- Run at least three fixed seeds for the principal result.
- Define the success criterion before looking at the final test results; do not retrofit a metric to match the historical "2%" claim.

### Acceptance criteria

- A CPU smoke run completes and emits a checkpoint plus structured metrics.
- A GPU run on the workstation completes from a documented command.
- The same seed/config reproduces the same generated evaluation set and materially consistent metrics.
- Training and test metrics are clearly separated.
- The result artifact includes all provenance required to re-run it.
- The checkpoint contains projection state plus metadata rather than a redundant frozen GPT-2 copy.
- Historical numeric-alignment claims are either reproduced under a defined metric or labeled superseded/not reproduced.

Scientific failure does not block CP6 if the pipeline is valid and the negative result is documented.


## CP6 closure evidence

Store the canonical commands, metric definitions, seeded result table, and artifact paths in `docs/checkpoints/CP06_NUMERIC_ALIGNMENT.md`.

---

## CP7 - Text-to-numeric validated experiment


## Objective

Validate text-token representations as inputs to the numeric output projection and quantify whether Stage-1 initialization helps.


## Deliverables

- Use attention-aware pooling for variable-length tokenized numeric strings.
- Evaluate two initializations: fresh output projection and output projection transferred from CP6.
- Keep the transformer frozen for the primary experiment.
- Add in-range evaluation and clearly separated out-of-range/format robustness sets.
- Include integer, decimal, varying precision, and representative numeric formatting variants.
- Use fixed seeds and structured run metadata identical to CP6.
- Define a metric before final test evaluation.
- Record whether CP6 transfer helps, hurts, or has no measurable effect.

### Acceptance criteria

- CPU smoke and GPU training/evaluation commands complete.
- Padding invariance is covered by tests.
- Fresh-vs-transferred initialization is reported under identical data and seeds.
- Results are replicated across the predefined seeds.
- In-range and robustness/OOD metrics are reported separately.
- README does not state a text-to-number error rate unless it points to the reproducible CP7 result.


## CP7 closure evidence

Document the experiment design, initialization comparison, robustness sets, and results in `docs/checkpoints/CP07_TEXT_ALIGNMENT.md`.

---

## CP8 - Correct mixed text/numeric embedding architecture


## Objective

Implement the core mixed-input mechanism correctly rather than concatenating hidden dimensions.


## Design

Parse each input around `$$value&&` markers.
Tokenize ordinary text segments with the base tokenizer.
Project each numeric value from scalar to `hidden_size`.
Insert that projected vector as one sequence embedding at the exact marker position.
Pad the resulting embedding sequences across the batch and construct a matching attention mask.
Run the frozen transformer on `inputs_embeds`.
Pool the last valid hidden state and map it through the numeric output projection.

This preserves the transformer's required hidden dimension while giving each numeric value a first-class sequence position.

### Deliverables

- Replace the current mixed branch in `NumericLMWrapper`.
- Preserve batch boundaries.
- Support zero, one, or multiple numeric markers according to an explicit policy.
- Make input and output projections trainable by default for the mixed task while keeping the transformer frozen.
- Define behavior for malformed markers.
- Add tests for batch size 1, variable batch size, variable text lengths, multiple numbers, negative/decimal values, and padding.
- Remove the unused/collapsed `_process_mixed_input` behavior.
- Add an embedding-construction unit test that verifies numeric vectors occupy the intended sequence positions.


## Acceptance criteria

- No tensor passed to GPT-2 has a hidden dimension different from `model.config.hidden_size`.
- Batch size is preserved end to end.
- Multiple markers in one sample produce multiple numeric sequence positions.
- A mixed forward pass returns `[batch, 1]`.
- A one-step mixed training test produces finite loss and gradients in the intended projection layers.
- Transformer weights remain frozen in the default mixed configuration.


## CP8 closure evidence

Document the sequence construction algorithm and tests in `docs/checkpoints/CP08_MIXED_ARCHITECTURE.md`.

---

## CP9 - Minimum-number mixed-task experiment


## Objective

Use the repaired mixed architecture on the repository's actual algorithmic task: selecting the minimum from text containing one or more numeric values.


## Deliverables

- Regenerate deterministic minimum-task train/validation/test data through CP5.
- Hold out some wording templates or controlled combinations to measure language-template generalization rather than memorization.
- Train the mixed numeric-aware model with the transformer frozen as the primary condition.
- Build a text-only frozen-transformer linear-probe baseline using the same examples.
- Keep a trivial programmatic oracle only as a data-integrity reference, not as a model comparison.
- Report error by number of values in the list, not only an aggregate metric.
- Evaluate multiple seeds.
- If desired, add transformer fine-tuning/LoRA only as a secondary labeled experiment after the frozen-transformer result is established.


## Acceptance criteria

- No train/test template leakage under the declared split.
- Mixed and text-only baselines use identical test data and metric definitions.
- Results include aggregate MAE/RMSE plus stratification by list length.
- Failure cases are sampled and categorized.
- The model result is reproducible from one documented command plus a configuration file.
- The experiment answers whether explicit numeric embeddings add measurable value on this task, without claiming more than the evidence supports.

### CP9 closure evidence

Write the protocol, baselines, split logic, seeded results, and failure analysis in `docs/checkpoints/CP09_MINIMUM_TASK.md`.

---

## CP10 - Reproducible benchmark and experiment system


## Objective

Replace the legacy spreadsheets with machine-readable, rerunnable benchmark artifacts and scale the validated experiments to the intended models.


## Deliverables

- Add unified `train` and `evaluate` entry points with task/config selection.
- Write one structured run directory per experiment containing config, environment, metrics, logs, and checkpoint metadata.
- Record Git SHA, dirty-state flag, model identifier/revision, seed, Python/package versions, GPU name, elapsed time, throughput, and peak VRAM.
- Convert the old Excel sheets to clearly labeled legacy CSV/Markdown observations.
- Benchmark GPT-2 and GPT-2 Large using the finalized architecture.
- Repeat the batch-size/memory study under controlled settings.
- Use the two RTX 3090 GPUs explicitly by device selection; distributed training is unnecessary unless a validated workload needs it.
- Produce consolidated CSV/JSON result tables and plots from structured run data.
- Compare CP6, CP7, and CP9 outcomes without mixing incompatible metrics.

### Acceptance criteria

- A single documented command can reproduce each canonical benchmark configuration.
- Every result row links back to an immutable config and checkpoint/run ID.
- Repeated runs with the same seed/config have documented reproducibility tolerance.
- Timing and VRAM measurements are collected consistently.
- Historical Excel observations are preserved but visually separated from new reproducible results.
- README performance statements refer only to CP10 artifacts.
- Any failure to reproduce the historical 2%/25% figures is reported plainly.


## CP10 closure evidence

Store benchmark methodology, environment, canonical run IDs, result tables, and legacy comparison in `docs/checkpoints/CP10_BENCHMARKS.md`.

---

## CP11 - CI, documentation, packaging, and release acceptance


## Objective

Make the repository maintainable by someone other than the original experiment author and close the project as a reproducible research release.


## Deliverables

- Add GitHub Actions for supported Python versions and offline CPU tests.
- Run syntax/static checks and pytest on push and pull request.
- Finalize `pyproject.toml`, package metadata, versioning, and console entry points.
- Rewrite README around architecture, installation, tasks, reproducibility, evaluation, results, and limitations.
- Add an architecture diagram or precise sequence-construction description.
- Add license/citation metadata if appropriate for the intended public use.
- Add a reproducibility checklist and release checklist.
- Remove or archive superseded scripts only after compatibility commands are documented.
- Ensure all CP2-CP10 closure documents exist and agree with the code.
- Prepare a versioned release candidate locally; pushing/tagging remains an explicit repository-owner action.

### Final acceptance criteria

From a fresh clone:

```text
1. create the documented Python environment
2. install the project
3. run offline unit/smoke tests
4. generate a small deterministic dataset
5. run N->N smoke training/evaluation
6. run text->N smoke training/evaluation
7. run mixed minimum-task smoke training/evaluation
8. load each produced checkpoint and re-evaluate it
9. generate a structured results summary
```

All steps must succeed without editing source files.

CI must be green.
Documentation must distinguish historical observations from reproduced results.
No known critical or high-severity audit item may remain unresolved or undocumented as an intentional limitation.

When those conditions are met, the repository can be called complete for its stated research scope.

## Planned status table

| CP | Name | Current status |
| --- | --- | --- |
| CP1 | Stabilization | Complete: syntax/config only |
| CP2 | Environment/dependencies | Complete: Python 3.12 + CUDA 12.6 env |
| CP3 | Core contracts/checkpoints | Complete: package contracts + schema-v1 checkpoints |
| CP4 | Automated tests | Complete: offline pytest suite |
| CP5 | Deterministic data | Complete: seeded generators + manifests + validation |
| CP6 | Numeric alignment validation | Complete: dataset-backed numeric run pipeline |
| CP7 | Text alignment validation | Complete: text-to-numeric pipeline + transfer init |
| CP8 | Mixed architecture | Complete: sequence-inserted numeric embeddings |
| CP9 | Minimum-task experiment | Planned |
| CP10 | Reproducible benchmarks | Planned |
| CP11 | Release hardening | Planned |

## Execution rule

Do not advance a checkpoint merely because code was written.
A checkpoint closes only when its acceptance criteria have been executed and its closure evidence is written locally.
