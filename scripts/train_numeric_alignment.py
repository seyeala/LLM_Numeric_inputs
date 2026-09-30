"""Train CP6 numeric-to-numeric alignment from a CP5 manifest."""

from __future__ import annotations

import argparse

from llm_numeric_inputs.experiments.numeric_alignment import (
    add_numeric_experiment_args,
    load_numeric_experiment_config,
    train_numeric_alignment,
)


def build_parser() -> argparse.ArgumentParser:
    return add_numeric_experiment_args(argparse.ArgumentParser(description=__doc__))


def main() -> None:
    args = build_parser().parse_args()
    run_dir = train_numeric_alignment(
        load_numeric_experiment_config(args.config),
        args.manifest,
        args.output_dir,
    )
    print(f"CP6 numeric run directory: {run_dir}")


if __name__ == "__main__":
    main()
