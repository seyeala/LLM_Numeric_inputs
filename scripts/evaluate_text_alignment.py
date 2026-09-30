"""Evaluate a CP7 text-to-numeric checkpoint on a CP5 manifest split."""

from __future__ import annotations

import argparse

from llm_numeric_inputs.devices import resolve_device
from llm_numeric_inputs.experiments.run_metadata import write_json
from llm_numeric_inputs.experiments.text_alignment import build_text_model_from_checkpoint
from llm_numeric_inputs.experiments.text_evaluation import evaluate_text_alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="CP7 schema checkpoint path.")
    parser.add_argument("--manifest", required=True, help="CP5 text_to_numeric manifest path.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = build_text_model_from_checkpoint(args.checkpoint, device)
    metrics = evaluate_text_alignment(model, model.tokenizer, args.manifest, args.split, args.batch_size, device)
    if args.output:
        write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
