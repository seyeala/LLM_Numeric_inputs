"""Backward-compatible imports for pre-CP3 code.

New code should import from llm_numeric_inputs directly.
"""

from llm_numeric_inputs.checkpointing import load_specific_weights
from llm_numeric_inputs.devices import clear_cuda_memory, print_cuda_memory
from llm_numeric_inputs.wrapper import NumericLMWrapper

__all__ = ["NumericLMWrapper", "clear_cuda_memory", "load_specific_weights", "print_cuda_memory"]
