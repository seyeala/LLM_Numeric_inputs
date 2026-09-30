"""Collect structured run artifacts into benchmark JSON/CSV tables."""

from __future__ import annotations

import argparse

from llm_numeric_inputs.experiments.benchmarks import collect_and_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs", help="Root directory containing run_manifest.json files.")
    parser.add_argument("--output-dir", default="benchmarks/generated", help="Output directory for summary tables.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = collect_and_write(args.runs_root, args.output_dir)
    print(result)


if __name__ == "__main__":
    main()
