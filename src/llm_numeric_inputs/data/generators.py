"""Pure deterministic record generators for numeric-input tasks."""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np

from llm_numeric_inputs.data.schemas import TaskName

DESCRIPTION_TEMPLATES = [
    "Among the numbers ",
    "Considering the values ",
    "When comparing ",
    "Looking at the sequence ",
    "Given the numbers ",
    "Among these figures ",
    "From the list ",
    "Observing the numbers ",
    "In the collection ",
    "Considering the series ",
    "From this set ",
    "Taking into account ",
    "Given the array ",
    "Looking over ",
    "Reviewing the numbers ",
    "Within the group ",
    "From the array ",
    "Considering this selection ",
    "Examining the values ",
    "Analyzing the set ",
    "Among the chosen numbers ",
    "Selecting from ",
    "Highlighting the numbers ",
    "Given the selection ",
    "From these options ",
    "Among this collection ",
    "Considering these figures ",
    "From the numbers ",
    "Evaluating the set ",
    "Taking a look at ",
]

QUESTION_VARIATIONS = [
    "what is the minimum number?",
    "can you identify the smallest value?",
    "which number is the least?",
    "find the lowest number.",
    "what's the smallest number among these?",
    "identify the minimum number.",
    "which of these is the smallest?",
    "determine the smallest value.",
    "which is the least among them?",
    "spot the minimum number.",
    "select the smallest number.",
    "pick out the smallest value.",
    "highlight the least number.",
    "which number is the lowest?",
    "locate the smallest number.",
    "distinguish the minimum value.",
    "choose the smallest number.",
    "point out the lowest number.",
    "decipher the smallest number.",
    "what is the least number?",
]


def _format_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _rngs(seed: int) -> tuple[np.random.Generator, random.Random]:
    return np.random.default_rng(seed), random.Random(seed)


def generate_numeric_to_numeric_records(
    count: int,
    seed: int,
    min_val: float = 0.0,
    max_val: float = 100.0,
    decimals: int = 4,
) -> Iterator[dict[str, str]]:
    rng, _ = _rngs(seed)
    for value in rng.uniform(min_val, max_val, count):
        text = _format_number(float(value), decimals)
        yield {"input": text, "output": text}


def generate_text_to_numeric_records(
    count: int,
    seed: int,
    min_val: float = 0.0,
    max_val: float = 100.0,
    decimals: int = 4,
) -> Iterator[dict[str, str]]:
    for record in generate_numeric_to_numeric_records(count, seed, min_val, max_val, decimals):
        yield {"input": f"$${record['input']}&&", "output": record["output"]}


def generate_minimum_records(
    count: int,
    seed: int,
    min_val: float = 0.0,
    max_val: float = 100.0,
    decimals: int = 2,
    min_terms: int = 1,
    max_terms: int = 10,
) -> Iterator[dict[str, str]]:
    if min_terms <= 0 or max_terms < min_terms:
        raise ValueError("minimum-task term bounds must satisfy 0 < min_terms <= max_terms.")
    np_rng, py_rng = _rngs(seed)
    for _ in range(count):
        num_terms = int(np_rng.integers(min_terms, max_terms + 1))
        raw_values = np_rng.uniform(min_val, max_val, num_terms)
        shown_values = [_format_number(float(value), decimals) for value in raw_values]
        answer = min(shown_values, key=lambda text: float(text))
        numbers_text = " and ".join(f"$${value}&&" for value in shown_values)
        prompt = f"{_choice(py_rng, DESCRIPTION_TEMPLATES)}{numbers_text}, {_choice(py_rng, QUESTION_VARIATIONS)}"
        yield {"input": prompt, "output": answer}


def generate_records(
    task: TaskName | str,
    count: int,
    seed: int,
    min_val: float = 0.0,
    max_val: float = 100.0,
    numeric_decimals: int = 4,
    minimum_decimals: int = 2,
    min_terms: int = 1,
    max_terms: int = 10,
) -> Iterator[dict[str, str]]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if min_val >= max_val:
        raise ValueError("min_val must be smaller than max_val.")
    task_name = TaskName(task)
    if task_name == TaskName.NUMERIC_TO_NUMERIC:
        yield from generate_numeric_to_numeric_records(count, seed, min_val, max_val, numeric_decimals)
    elif task_name == TaskName.TEXT_TO_NUMERIC:
        yield from generate_text_to_numeric_records(count, seed, min_val, max_val, numeric_decimals)
    elif task_name == TaskName.MINIMUM:
        yield from generate_minimum_records(count, seed, min_val, max_val, minimum_decimals, min_terms, max_terms)
    else:
        raise ValueError(f"Unsupported task: {task!r}")
