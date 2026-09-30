"""Dataset-backed mixed minimum-number training and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam, lr_scheduler

from llm_numeric_inputs.checkpointing import load_checkpoint, load_checkpoint_into_model, save_checkpoint
from llm_numeric_inputs.config import TrainConfig, train_config_from_mapping
from llm_numeric_inputs.devices import resolve_device
from llm_numeric_inputs.experiments.minimum_data import iter_minimum_batches, read_minimum_split
from llm_numeric_inputs.experiments.minimum_evaluation import evaluate_minimum_alignment
from llm_numeric_inputs.experiments.run_metadata import environment_metadata, new_run_id, write_json
from llm_numeric_inputs.experiments.tiny_model import TINY_MODEL_NAME, TinyCausalLM, TinyTokenizer
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.seeding import set_seed
from llm_numeric_inputs.wrapper import NumericLMWrapper


@dataclass(slots=True)
class MinimumExperimentConfig:
    train: TrainConfig
    validation_split: str = "validation"
    test_split: str = "test"
    run_id: str | None = None
    tiny_hidden_size: int = 8
    tiny_n_positions: int = 64

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train"] = self.train.to_dict()
        return payload


def load_minimum_experiment_config(path: str | Path) -> MinimumExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    train = train_config_from_mapping(dict(raw.get("train", raw)))
    return MinimumExperimentConfig(
        train=train,
        validation_split=raw.get("validation_split", "validation"),
        test_split=raw.get("test_split", "test"),
        run_id=raw.get("run_id"),
        tiny_hidden_size=int(raw.get("tiny_hidden_size", 8)),
        tiny_n_positions=int(raw.get("tiny_n_positions", 64)),
    )


def build_minimum_model(config: MinimumExperimentConfig, device: torch.device) -> NumericLMWrapper:
    model_name = config.train.model_name
    model = None
    tokenizer = None
    if model_name == TINY_MODEL_NAME:
        model = TinyCausalLM(config.tiny_hidden_size, config.tiny_n_positions)
        tokenizer = TinyTokenizer()
    wrapper = NumericLMWrapper(
        model_name, mode=InputMode.MIXED_TO_NUMERIC, device=device, model=model, tokenizer=tokenizer
    )
    wrapper.configure_trainable_layers(True, True, False)
    if wrapper.tokenizer is not None and getattr(wrapper.tokenizer, "pad_token", None) is None:
        wrapper.tokenizer.pad_token = wrapper.tokenizer.eos_token
    return wrapper


def _train_one_epoch(
    model: torch.nn.Module,
    texts: list[str],
    targets: torch.Tensor,
    config: TrainConfig,
    optimizer: Adam,
    scaler: GradScaler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    batches = 0
    for batch_texts, batch_targets in iter_minimum_batches(texts, targets, config.batch_size):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model({"input_text": batch_texts})
            loss = nn.MSELoss()(outputs, batch_targets.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach().cpu().item())
        batches += 1
    return {"train_mse_loss": loss_sum / max(batches, 1), "train_batches": float(batches)}


def train_minimum_alignment(
    config: MinimumExperimentConfig,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Train the mixed minimum-task model and write a structured run directory."""
    set_seed(config.train.seed, deterministic=config.train.deterministic)
    device = resolve_device(config.train.device)
    run_id = config.run_id or new_run_id("cp9_minimum")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "resolved_config.json", config.to_dict())
    write_json(run_dir / "environment_report.json", environment_metadata())

    model = build_minimum_model(config, device)
    if config.train.model_path_load:
        load_checkpoint_into_model(model, config.train.model_path_load, strict=False)

    train_texts, train_targets, manifest = read_minimum_split(manifest_path, "train")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters are enabled for CP9 minimum training.")
    optimizer = Adam(trainable, lr=config.train.lr)
    scheduler = (
        lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.scheduler_step_size,
            gamma=config.train.scheduler_gamma,
        )
        if config.train.shl
        else None
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    epoch_metrics: list[dict[str, Any]] = []
    for epoch in range(config.train.num_epochs):
        train_metrics = _train_one_epoch(model, train_texts, train_targets, config.train, optimizer, scaler, device)
        validation_metrics = evaluate_minimum_alignment(
            model,
            manifest_path,
            config.validation_split,
            config.train.batch_size,
            device,
        )
        epoch_metrics.append({"epoch": epoch + 1, **train_metrics, "validation": validation_metrics})
        if scheduler:
            scheduler.step()

    test_metrics = evaluate_minimum_alignment(model, manifest_path, config.test_split, config.train.batch_size, device)
    checkpoint_path = run_dir / "checkpoint.pth"
    save_checkpoint(
        model,
        checkpoint_path,
        InputMode.MIXED_TO_NUMERIC,
        train_config=config.train,
        metadata={"cp": "CP9", "run_id": run_id},
    )
    write_json(run_dir / "epoch_metrics.json", epoch_metrics)
    write_json(run_dir / "test_metrics.json", test_metrics)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "checkpoint": str(checkpoint_path),
            "dataset_manifest": str(Path(manifest_path)),
            "dataset_task": manifest["task"],
            "dataset_seed": manifest["seed"],
            "validation_split": config.validation_split,
            "test_split": config.test_split,
            "test_metrics": test_metrics,
        },
    )
    return run_dir


def build_minimum_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> NumericLMWrapper:
    payload = load_checkpoint(checkpoint_path, map_location=device)
    train_config = train_config_from_mapping(payload.get("train_config") or {})
    exp_config = MinimumExperimentConfig(train=train_config)
    model = build_minimum_model(exp_config, device)
    load_checkpoint_into_model(model, checkpoint_path, strict=False)
    return model


def add_minimum_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="CP9 minimum experiment YAML config.")
    parser.add_argument("--manifest", required=True, help="CP5 minimum dataset manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory where the run directory will be created.")
    return parser
