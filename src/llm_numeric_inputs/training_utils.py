"""Shared lightweight training utilities for compatibility scripts."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam, lr_scheduler

from llm_numeric_inputs.config import TrainConfig
from llm_numeric_inputs.devices import print_cuda_memory


def generate_numeric_batch(config: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand(config.batch_size, 1, device=device) * (config.max_val - config.min_val) + config.min_val
    return inputs, inputs


def generate_text_numeric_batch(
    config: TrainConfig, device: torch.device, tokenizer
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    values = torch.rand(config.batch_size, 1) * (config.max_val - config.min_val) + config.min_val
    text_inputs = [f"$${number.item()}&&" for number in values]
    tokenized = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    return tokenized, values.to(device)


def generate_mixed_numeric_batch(
    config: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, list[str]], torch.Tensor]:
    """Generate mixed text/numeric-marker inputs and numeric targets."""
    values = torch.rand(config.batch_size, 1) * (config.max_val - config.min_val) + config.min_val
    text_inputs = [f"The measured value is $${number.item()}&&." for number in values]
    return {"input_text": text_inputs}, values.to(device)


def train_projection_epochs(
    llm: torch.nn.Module,
    config: TrainConfig,
    batch_factory: Callable[[], tuple[object, torch.Tensor]],
) -> None:
    device = next(llm.parameters()).device
    trainable_params = [param for param in llm.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters are enabled.")
    optimizer = Adam(trainable_params, lr=config.lr)
    scheduler = (
        lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        if config.shl
        else None
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    llm.train()
    for epoch in range(config.num_epochs):
        start = time.time()
        loss_sum = 0.0
        for _ in range(config.num_batches):
            batch_inputs, batch_targets = batch_factory()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = llm(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)
                loss_sum += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler:
            scheduler.step()
        print(
            f"Epoch {epoch + 1}: Average Loss = {loss_sum / config.num_batches:.4f}, "
            f"Total Compute Time = {time.time() - start:.2f} seconds"
        )
        print_cuda_memory()
