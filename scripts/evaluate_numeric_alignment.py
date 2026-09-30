"""Evaluate a CP6 numeric-to-numeric checkpoint on a CP5 manifest split."""

from __future__ import annotations

import argparse

from llm_numeric_inputs.devices import resolve_device
from llm_numeric_inputs.experiments.evaluation import evaluate_numeric_alignment
from llm_numeric_inputs.experiments.numeric_alignment import build_model_from_checkpoint
from llm_numeric_inputs.experiments.run_metadata import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="CP6 schema checkpoint path.")
    parser.add_argument("--manifest", required=True, help="CP5 numeric_to_numeric manifest path.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = build_model_from_checkpoint(args.checkpoint, device)
    metrics = evaluate_numeric_alignment(model, args.manifest, args.split, args.batch_size, device)
    if args.output:
        write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
