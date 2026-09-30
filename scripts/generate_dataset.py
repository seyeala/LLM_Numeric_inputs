"""Generate deterministic JSONL datasets for numeric-input tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_numeric_inputs.data.dataset import generate_dataset
from llm_numeric_inputs.data.schemas import TaskName


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=[task.value for task in TaskName])
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-val", type=float, default=0.0)
    parser.add_argument("--max-val", type=float, default=100.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--numeric-decimals", type=int, default=4)
    parser.add_argument("--minimum-decimals", type=int, default=2)
    parser.add_argument("--min-terms", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=10)
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = generate_dataset(
        task=args.task,
        count=args.count,
        seed=args.seed,
        output_dir=args.output_dir,
        min_val=args.min_val,
        max_val=args.max_val,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        numeric_decimals=args.numeric_decimals,
        minimum_decimals=args.minimum_decimals,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
        validate=not args.skip_validation,
    )
    print(f"Generated dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
