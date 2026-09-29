import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, lr_scheduler
import yaml

from wrapperNM import NumericLMWrapper, clear_cuda_memory, load_specific_weights, print_cuda_memory


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def apply_overrides(config, args):
    for key in ["num_epochs", "num_batches", "batch_size", "lr", "min_val", "max_val", "model_name", "device", "model_path_load", "model_path_save", "shl"]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    config.setdefault("num_epochs", 2)
    config.setdefault("num_batches", 10)
    config.setdefault("batch_size", 8)
    config.setdefault("lr", 0.003)
    config.setdefault("min_val", 0.0)
    config.setdefault("max_val", 100.0)
    config.setdefault("model_name", "openai-community/gpt2")
    config.setdefault("device", "auto")
    config.setdefault("shl", False)
    config.setdefault("model_path_load", None)
    config.setdefault("model_path_save", "./chk/trained_numeric_lm.pth")
    return config


def resolve_device(requested):
    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def ensure_parent_dir(path):
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def generate_data(batch_size, min_val, max_val, device):
    """Generate random numeric inputs and identical numeric targets."""
    inputs = torch.rand(batch_size, 1, device=device) * (max_val - min_val) + min_val
    return inputs, inputs


def alignment(llm, config, num_epochs, min_val, max_val, model_path_load, model_path_save, shl):
    llm.train()
    device = next(llm.parameters()).device
    optimizer = Adam(filter(lambda p: p.requires_grad, llm.parameters()), lr=float(config["lr"]))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
    scaler = GradScaler(enabled=device.type == "cuda")

    if model_path_load:
        try:
            print(f"Attempting to load model from {model_path_load}")
            load_specific_weights(llm, model_path_load)
            print(f"Successfully loaded model from {model_path_load}")
        except FileNotFoundError:
            print(f"No model found at {model_path_load}, starting from scratch.")
        except RuntimeError as error:
            print(f"Failed to load model. Error: {error}")
            print("Check if the model architecture during saving matches the one during loading.")

    num_batches = int(config["num_batches"])
    batch_size = int(config["batch_size"])

    for epoch in range(int(num_epochs)):
        total_cpu_start = time.time()
        cumulative_data_cpu_time = 0.0
        epoch_loss_sum = 0.0

        for _ in range(num_batches):
            data_cpu_start = time.time()
            batch_inputs, batch_targets = generate_data(batch_size, min_val, max_val, device)
            cumulative_data_cpu_time += time.time() - data_cpu_start

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                outputs = llm(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)
                epoch_loss_sum += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        average_loss = epoch_loss_sum / num_batches
        print(f"End of Epoch {epoch + 1}")
        print(f"Total Compute Time for Epoch: {time.time() - total_cpu_start:.2f} seconds")
        print(f"Cumulative Data Loading CPU Time: {cumulative_data_cpu_time:.2f} seconds")
        print(f"Average Loss for Epoch: {average_loss:.4f}")
        if scheduler:
            scheduler.step()
        print_cuda_memory()

    ensure_parent_dir(model_path_save)
    torch.save(llm.state_dict(), model_path_save)
    print(f"Saved trained model to {model_path_save}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train numeric-to-numeric projection alignment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--num_batches", type=int, help="Number of batches per epoch.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--min_val", type=float, help="Minimum generated numeric value.")
    parser.add_argument("--max_val", type=float, help="Maximum generated numeric value.")
    parser.add_argument("--model_name", type=str, help="Hugging Face model name or local model path.")
    parser.add_argument("--model_path_load", type=str, help="Checkpoint path to load before training.")
    parser.add_argument("--model_path_save", type=str, help="Checkpoint path to save after training.")
    parser.add_argument("--device", type=str, help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--shl", type=str2bool, nargs="?", const=True, help="Whether to use StepLR scheduler.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)

    device = resolve_device(config["device"])
    clear_cuda_memory()
    numeric_lm = NumericLMWrapper(
        config["model_name"],
        project_input=True,
        project_output=True,
        device=device,
    )
    numeric_lm.configure_trainable_layers(
        train_input_projection=True,
        train_output_projection=True,
        train_transformer=False,
    )

    alignment(
        numeric_lm,
        config,
        config["num_epochs"],
        config["min_val"],
        config["max_val"],
        config.get("model_path_load"),
        config["model_path_save"],
        bool(config.get("shl", False)),
    )
