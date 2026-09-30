"""Train CP7 text-to-numeric alignment from a CP5 manifest."""

from __future__ import annotations

import argparse

from llm_numeric_inputs.experiments.text_alignment import (
    add_text_experiment_args,
    load_text_experiment_config,
    train_text_alignment,
)


def build_parser() -> argparse.ArgumentParser:
    return add_text_experiment_args(argparse.ArgumentParser(description=__doc__))


def main() -> None:
    args = build_parser().parse_args()
    run_dir = train_text_alignment(
        load_text_experiment_config(args.config),
        args.manifest,
        args.output_dir,
    )
    print(f"CP7 text run directory: {run_dir}")


if __name__ == "__main__":
    main()
