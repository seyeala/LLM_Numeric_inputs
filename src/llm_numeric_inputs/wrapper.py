"""Numeric language-model wrapper with explicit CP3 contracts."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_numeric_inputs.modes import InputMode, NumericSequenceStrategy
from llm_numeric_inputs.pooling import last_valid_token_pool


def _hidden_size(config: Any) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AttributeError("Model config does not expose hidden_size, n_embd, or d_model.")


def _n_positions(config: Any) -> int:
    for name in ("n_positions", "max_position_embeddings", "max_sequence_length"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return 1


def _legacy_mode(project_input: bool, project_output: bool, mixed_input: bool) -> InputMode | None:
    if mixed_input:
        return InputMode.MIXED_TO_NUMERIC
    if project_input and project_output:
        return InputMode.NUMERIC_TO_NUMERIC
    if project_output and not project_input:
        return InputMode.TEXT_TO_NUMERIC
    return None


class NumericLMWrapper(nn.Module):
    """Wrap a causal LM with scalar numeric input/output projections.

    Formal CP3 contracts:
    - numeric_to_numeric: Tensor [B, 1] -> Tensor [B, 1]
    - text_to_numeric: tokenized batch dict -> Tensor [B, 1]
    - mixed_to_numeric: deferred to CP8 and raises NotImplementedError.
    """

    def __init__(
        self,
        model_name: str,
        mode: InputMode | str | None = None,
        project_input: bool = False,
        project_output: bool = False,
        mixed_input: bool = False,
        device: str | torch.device = "cpu",
        numeric_sequence_strategy: NumericSequenceStrategy | str = NumericSequenceStrategy.SINGLE_TOKEN,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = None
        self.model = model if model is not None else AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)

        resolved_mode = (
            InputMode(mode) if mode is not None else _legacy_mode(project_input, project_output, mixed_input)
        )
        self.mode = resolved_mode
        self.project_input = project_input or resolved_mode in {
            InputMode.NUMERIC_TO_NUMERIC,
            InputMode.MIXED_TO_NUMERIC,
        }
        self.project_output = project_output or resolved_mode in {
            InputMode.NUMERIC_TO_NUMERIC,
            InputMode.TEXT_TO_NUMERIC,
            InputMode.MIXED_TO_NUMERIC,
        }
        self.mixed_input = mixed_input or resolved_mode == InputMode.MIXED_TO_NUMERIC
        self.numeric_sequence_strategy = NumericSequenceStrategy(numeric_sequence_strategy)

        embedding_dim = _hidden_size(self.model.config)
        self.input_projection = nn.Linear(1, embedding_dim).to(self.device)
        self.output_projection = nn.Linear(embedding_dim, 1).to(self.device)

    def _model_hidden_states(self, **kwargs):
        outputs = self.model(**kwargs, return_dict=True, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Base model did not return hidden states.")
        return hidden_states[-1]

    def forward_numeric(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map numeric tensor [B, 1] to numeric tensor [B, 1]."""
        if not torch.is_tensor(inputs):
            raise TypeError("numeric inputs must be a torch.Tensor.")
        if inputs.ndim != 2 or inputs.shape[1] != 1:
            raise ValueError(f"numeric inputs must have shape [batch, 1]; got {tuple(inputs.shape)}")
        numeric = inputs.to(self.device)
        projected = self.input_projection(numeric)
        if self.numeric_sequence_strategy == NumericSequenceStrategy.SINGLE_TOKEN:
            inputs_embeds = projected.unsqueeze(1)
            attention_mask = torch.ones((numeric.shape[0], 1), dtype=torch.long, device=self.device)
        else:
            sequence_length = _n_positions(self.model.config)
            inputs_embeds = projected.unsqueeze(1).expand(-1, sequence_length, -1)
            attention_mask = torch.ones((numeric.shape[0], sequence_length), dtype=torch.long, device=self.device)
        hidden = self._model_hidden_states(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = last_valid_token_pool(hidden, attention_mask)
        return self.output_projection(pooled)

    def forward_text(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Map tokenized text batch to numeric tensor [B, 1]."""
        if not isinstance(inputs, dict):
            raise TypeError("text inputs must be a tokenizer dictionary.")
        moved = {key: value.to(self.device) for key, value in inputs.items() if torch.is_tensor(value)}
        if "input_ids" not in moved:
            raise ValueError("text inputs must include input_ids.")
        if "attention_mask" not in moved:
            moved["attention_mask"] = torch.ones_like(moved["input_ids"], dtype=torch.long, device=self.device)
        hidden = self._model_hidden_states(**moved)
        pooled = last_valid_token_pool(hidden, moved["attention_mask"])
        return self.output_projection(pooled)

    def forward_mixed(self, inputs: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError("mixed_to_numeric is deferred to CP8; current mixed code is not a validated path.")

    def forward(self, inputs):
        if self.mode == InputMode.NUMERIC_TO_NUMERIC:
            return self.forward_numeric(inputs)
        if self.mode == InputMode.TEXT_TO_NUMERIC:
            return self.forward_text(inputs)
        if self.mode == InputMode.MIXED_TO_NUMERIC:
            return self.forward_mixed(inputs)
        if isinstance(inputs, dict):
            moved = {key: value.to(self.device) for key, value in inputs.items() if torch.is_tensor(value)}
            outputs = self.model(**moved, return_dict=True, output_hidden_states=self.project_output)
            if self.project_output:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError("Base model did not return hidden states.")
                attention_mask = moved.get("attention_mask", torch.ones_like(moved["input_ids"], dtype=torch.long))
                return self.output_projection(last_valid_token_pool(hidden_states[-1], attention_mask))
            return outputs.logits if hasattr(outputs, "logits") else outputs
        raise ValueError("Wrapper mode is not set. Use an explicit InputMode for projected tasks.")

    def generate_text(self, input_text: str, **generate_kwargs) -> str:
        if self.mode is None and not self.project_input and not self.project_output:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids, **generate_kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raise NotImplementedError("Generate method is not implemented for projected input/output modes.")

    def configure_trainable_layers(
        self,
        train_input_projection: bool = True,
        train_output_projection: bool = True,
        train_transformer: bool = True,
    ) -> None:
        for param in self.input_projection.parameters():
            param.requires_grad = train_input_projection
        for param in self.output_projection.parameters():
            param.requires_grad = train_output_projection
        for param in self.model.parameters():
            param.requires_grad = train_transformer
