# Checkpoint closure notes

Each CP2–CP11 checkpoint closes with one evidence document in this directory.

A checkpoint is not complete merely because implementation code exists. Its acceptance commands must have been executed and the results recorded.

## Required sections

Every closure note should contain:

- checkpoint number and title,
- commit SHA(s),
- scope implemented,
- files added/changed,
- exact validation commands,
- validation results,
- environment information relevant to the run,
- known limitations or deferred items,
- acceptance-criteria checklist,
- next-checkpoint dependencies.

## Naming

Use the filenames defined in `docs/ROADMAP_CP2_CP11.md`, for example:

```text
CP02_ENVIRONMENT.md
CP03_CORE_CONTRACTS.md
CP04_TESTS.md
...
CP11_RELEASE.md
```

Do not copy historical performance claims into closure notes unless the checkpoint actually reproduces them under a defined metric.

For experiment checkpoints, include run IDs, seeds, model identifier/revision, data manifest, and result artifact paths.
