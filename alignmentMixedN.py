import argparse

from llm_numeric_inputs.config import add_common_train_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed text/numeric alignment placeholder.")
    add_common_train_args(parser)
    parser.add_argument(
        "--acknowledge_experimental",
        action="store_true",
        help="Acknowledge that mixed mode is deferred to CP8 and is not runnable in CP3.",
    )
    return parser


def main() -> None:
    build_parser().parse_args()
    raise SystemExit(
        "Mixed text-plus-numeric training is intentionally disabled in CP3. "
        "The current legacy architecture is invalid and will be replaced in CP8."
    )


if __name__ == "__main__":
    main()
