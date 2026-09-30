"""Evaluate a CP9 mixed minimum-number checkpoint on a CP5 manifest split."""

from __future__ import annotations

import argparse

from llm_numeric_inputs.devices import resolve_device
from llm_numeric_inputs.experiments.minimum_alignment import build_minimum_model_from_checkpoint
from llm_numeric_inputs.experiments.minimum_evaluation import evaluate_minimum_alignment
from llm_numeric_inputs.experiments.run_metadata import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="CP9 schema checkpoint path.")
    parser.add_argument("--manifest", required=True, help="CP5 minimum manifest path.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = build_minimum_model_from_checkpoint(args.checkpoint, device)
    metrics = evaluate_minimum_alignment(model, args.manifest, args.split, args.batch_size, device)
    if args.output:
        write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
