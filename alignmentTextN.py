import argparse

from llm_numeric_inputs.checkpointing import load_checkpoint_into_model, save_checkpoint
from llm_numeric_inputs.config import add_common_train_args, load_train_config
from llm_numeric_inputs.devices import resolve_device
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.seeding import set_seed
from llm_numeric_inputs.training_utils import generate_text_numeric_batch, train_projection_epochs
from llm_numeric_inputs.wrapper import NumericLMWrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train text-to-numeric projection alignment.")
    return add_common_train_args(parser)


def main() -> None:
    args = build_parser().parse_args()
    config = load_train_config(args.config, args)
    set_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)

    llm = NumericLMWrapper(config.model_name, mode=InputMode.TEXT_TO_NUMERIC, device=device)
    llm.configure_trainable_layers(
        train_input_projection=False,
        train_output_projection=True,
        train_transformer=False,
    )

    if llm.tokenizer.pad_token is None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token

    if config.model_path_load:
        print(f"Attempting to load checkpoint from {config.model_path_load}")
        load_checkpoint_into_model(llm, config.model_path_load, strict=False)
        print(f"Loaded checkpoint from {config.model_path_load}")

    train_projection_epochs(llm, config, lambda: generate_text_numeric_batch(config, device, llm.tokenizer))
    save_checkpoint(llm, config.model_path_save, InputMode.TEXT_TO_NUMERIC, train_config=config)
    print(f"Saved CP3 projection checkpoint to {config.model_path_save}")


if __name__ == "__main__":
    main()
