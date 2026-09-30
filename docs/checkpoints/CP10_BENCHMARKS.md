# CP10 - Reproducible benchmark aggregation

Date: 2026-09-29
Repository: `F:\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `e4f4076`

## Scope

CP10 adds a reproducible benchmark aggregation layer for structured run directories.
It does not invent benchmark results and does not claim that historical GPT-2/GPT-2 Large numbers were reproduced.

## Implementation

Added:

```text
src/llm_numeric_inputs/experiments/benchmarks.py
scripts/collect_benchmark_results.py
tests/test_benchmarks.py
```

The collector scans for `run_manifest.json` files below a run root and summarizes each run's config, environment, checkpoint, dataset reference, and test metrics.

## Outputs

Running:

```powershell
.\.venv\Scripts\python.exe scripts\collect_benchmark_results.py --runs-root runs --output-dir benchmarks\generated
```

writes:

```text
benchmarks/generated/benchmark_results.json
benchmarks/generated/benchmark_results.csv
```

These generated outputs are ignored by Git. Re-run the collector to regenerate them from local run artifacts.

## Local collection result

The CP10 collector found 7 local run directories during implementation. These include smoke and tiny-model runs from CP6, CP7, and CP9.

## Legacy benchmark policy

The old Excel files in `Benchmarks_training/` remain historical observations. They are not treated as reproduced CP10 benchmark results because they do not contain enough provenance for reproducibility.

## Validation

CP10 is validated through the full repository test suite and dedicated benchmark aggregation tests.

```text
pytest includes tests/test_benchmarks.py
```

## Known limitations

- CP10 aggregates existing structured run artifacts; it does not itself launch expensive GPT-2/GPT-2 Large benchmark grids.
- Generated benchmark tables are local artifacts and are intentionally not committed.
- Historical spreadsheet observations remain separated from generated CP10 benchmark summaries.

## Next checkpoint

CP11 should add CI, release/reproducibility documentation, and final project closure evidence.
