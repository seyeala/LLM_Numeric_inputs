# Dataset directory

The large JSONL files in this directory are legacy generated artifacts from the pre-CP5 workflow.
They are retained for provenance and backward compatibility only.

Legacy files:

- `alignmentNN.jsonl`
- `alignmentTN.jsonl`
- `minimumalgo.jsonl`

New deterministic datasets should be generated with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task numeric_to_numeric --count 1000 --seed 42 --output-dir data\generated\numeric_to_numeric
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task text_to_numeric --count 1000 --seed 42 --output-dir data\generated\text_to_numeric
.\.venv\Scripts\python.exe scripts\generate_dataset.py --task minimum --count 1000 --seed 42 --output-dir data\generated\minimum
```

Each generated directory contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`.
Use `scripts\validate_dataset.py --manifest <path-to-manifest>` to validate records, row counts, and SHA-256 hashes.
