"""Validate a deterministic dataset manifest and referenced JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_numeric_inputs.data.dataset import manifest_summary
from llm_numeric_inputs.data.validation import validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = validate_manifest(args.manifest)
    print(f"Dataset validation ok: {manifest_summary(manifest)}")


if __name__ == "__main__":
    main()
