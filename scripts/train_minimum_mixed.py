"""Train CP9 mixed minimum-number alignment from a CP5 manifest."""

from __future__ import annotations

import argparse

from llm_numeric_inputs.experiments.minimum_alignment import (
    add_minimum_experiment_args,
    load_minimum_experiment_config,
    train_minimum_alignment,
)


def build_parser() -> argparse.ArgumentParser:
    return add_minimum_experiment_args(argparse.ArgumentParser(description=__doc__))


def main() -> None:
    args = build_parser().parse_args()
    run_dir = train_minimum_alignment(load_minimum_experiment_config(args.config), args.manifest, args.output_dir)
    print(f"CP9 minimum run directory: {run_dir}")


if __name__ == "__main__":
    main()
