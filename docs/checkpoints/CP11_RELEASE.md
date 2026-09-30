# CP11 - Release hardening

Date: 2026-09-29
Repository: `F:\LLM_Numeric_inputs`
Branch: `cp1-stabilization`

## Scope

CP11 closes the project roadmap with CI, release documentation, and reproducibility guidance.
It does not create new scientific claims.

## Added artifacts

```text
.github/workflows/tests.yml
docs/REPRODUCIBILITY.md
docs/RELEASE_CHECKLIST.md
docs/checkpoints/CP11_RELEASE.md
```

## CI coverage

The GitHub Actions workflow installs Python 3.12, installs the package with dev dependencies, runs Ruff, runs pytest offline, and runs `pip check`.

## Release guidance

`docs/REPRODUCIBILITY.md` explains environment setup, deterministic data generation, experiment execution, and benchmark aggregation.

`docs/RELEASE_CHECKLIST.md` defines the local release gate.

## Final validation

CP11 should be accepted only when the local full validation suite passes and the branch is pushed.

## Known limitations

- The public GitHub workflow will execute after push; this local run cannot assert remote CI success before GitHub runs it.
- Legacy large JSONL files remain in Git history.
- Full GPT-2/GPT-2 Large benchmark grids remain to be run by the repository owner when compute/time budget is allocated.

## Roadmap status

CP1 through CP11 are implemented as local commits on the `cp1-stabilization` branch.
