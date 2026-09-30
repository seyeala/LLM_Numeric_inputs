"""Dataset-backed numeric-to-numeric training and evaluation."""

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
from llm_numeric_inputs.experiments.evaluation import evaluate_numeric_alignment
from llm_numeric_inputs.experiments.numeric_data import iter_batches, read_numeric_split
from llm_numeric_inputs.experiments.run_metadata import environment_metadata, new_run_id, write_json
from llm_numeric_inputs.experiments.tiny_model import TINY_MODEL_NAME, TinyCausalLM
from llm_numeric_inputs.modes import InputMode
from llm_numeric_inputs.seeding import set_seed
from llm_numeric_inputs.wrapper import NumericLMWrapper


@dataclass(slots=True)
class NumericExperimentConfig:
    train: TrainConfig
    validation_split: str = "validation"
    test_split: str = "test"
    run_id: str | None = None
    tiny_hidden_size: int = 8
    tiny_n_positions: int = 4

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train"] = self.train.to_dict()
        return payload


def load_numeric_experiment_config(path: str | Path) -> NumericExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    train_raw = dict(raw.get("train", raw))
    train = train_config_from_mapping(train_raw)
    return NumericExperimentConfig(
        train=train,
        validation_split=raw.get("validation_split", "validation"),
        test_split=raw.get("test_split", "test"),
        run_id=raw.get("run_id"),
        tiny_hidden_size=int(raw.get("tiny_hidden_size", 8)),
        tiny_n_positions=int(raw.get("tiny_n_positions", 4)),
    )


def build_numeric_model(config: NumericExperimentConfig, device: torch.device) -> NumericLMWrapper:
    model_name = config.train.model_name
    model = None
    tokenizer = None
    if model_name == TINY_MODEL_NAME:
        model = TinyCausalLM(config.tiny_hidden_size, config.tiny_n_positions)
    wrapper = NumericLMWrapper(
        model_name,
        mode=InputMode.NUMERIC_TO_NUMERIC,
        device=device,
        numeric_sequence_strategy=config.train.numeric_sequence_strategy,
        model=model,
        tokenizer=tokenizer,
    )
    wrapper.configure_trainable_layers(True, True, False)
    return wrapper


def _train_one_epoch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: TrainConfig,
    optimizer: Adam,
    scaler: GradScaler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    batches = 0
    for batch_inputs, batch_targets in iter_batches(inputs, targets, config.batch_size):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(batch_inputs.to(device))
            loss = nn.MSELoss()(outputs, batch_targets.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach().cpu().item())
        batches += 1
    return {"train_mse_loss": loss_sum / max(batches, 1), "train_batches": float(batches)}


def train_numeric_alignment(
    config: NumericExperimentConfig,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Train numeric-to-numeric alignment and write a structured run directory."""
    set_seed(config.train.seed, deterministic=config.train.deterministic)
    device = resolve_device(config.train.device)
    run_id = config.run_id or new_run_id("cp6_numeric")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "resolved_config.json", config.to_dict())
    write_json(run_dir / "environment_report.json", environment_metadata())

    model = build_numeric_model(config, device)
    if config.train.model_path_load:
        load_checkpoint_into_model(model, config.train.model_path_load, strict=False)

    train_inputs, train_targets, manifest = read_numeric_split(manifest_path, "train")
    optimizer = Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=config.train.lr)
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
        train_metrics = _train_one_epoch(model, train_inputs, train_targets, config.train, optimizer, scaler, device)
        validation_metrics = evaluate_numeric_alignment(
            model,
            manifest_path,
            config.validation_split,
            config.train.batch_size,
            device,
        )
        payload = {"epoch": epoch + 1, **train_metrics, "validation": validation_metrics}
        epoch_metrics.append(payload)
        if scheduler:
            scheduler.step()
    test_metrics = evaluate_numeric_alignment(model, manifest_path, config.test_split, config.train.batch_size, device)
    checkpoint_path = run_dir / "checkpoint.pth"
    save_checkpoint(
        model,
        checkpoint_path,
        InputMode.NUMERIC_TO_NUMERIC,
        train_config=config.train,
        metadata={"cp": "CP6", "run_id": run_id},
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


def build_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> NumericLMWrapper:
    payload = load_checkpoint(checkpoint_path, map_location=device)
    train_config = train_config_from_mapping(payload.get("train_config") or {})
    exp_config = NumericExperimentConfig(train=train_config)
    model = build_numeric_model(exp_config, device)
    load_checkpoint_into_model(model, checkpoint_path, strict=False)
    return model


def add_numeric_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="CP6 numeric experiment YAML config.")
    parser.add_argument("--manifest", required=True, help="CP5 numeric_to_numeric dataset manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory where the run directory will be created.")
    return parser
