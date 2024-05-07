# -*- coding: utf-8 -*-
"""text_numeric_generator_for_LLM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oXIlWi0qDj_-aBotx6D-YdKanSyp3mg5
"""

import numpy as np
import random

# Generate a natural number n from 1 to 10
n = np.random.randint(1, 11)

# Generate n random float32 numbers
random_floats = np.random.rand(n).astype(np.float32) * 100  # Scaling to make numbers more varied

# Find the smallest number
answer_numeric = np.min(random_floats)

# List of 30 different templates for descriptions
descriptions = [
    "Among the numbers ", "Considering the values ", "When comparing ", "Looking at the sequence ",
    "Given the numbers ", "Among these figures ", "From the list ", "Observing the numbers ",
    "In the collection ", "Considering the series ", "From this set ", "Taking into account ",
    "Given the array ", "Looking over ", "Reviewing the numbers ", "Within the group ",
    "From the array ", "Considering this selection ", "Examining the values ", "Analyzing the set ",
    "Among the chosen numbers ", "Selecting from ", "Highlighting the numbers ", "Given the selection ",
    "From these options ", "Among this collection ", "Considering these figures ", "From the numbers ",
    "Evaluating the set ", "Taking a look at "
]

# List of 20 different variations of "what is the minimum number"
question_variations = [
    "what is the minimum number?", "can you identify the smallest value?", "which number is the least?",
    "find the lowest number.", "what's the smallest number among these?", "identify the minimum number.",
    "which of these is the smallest?", "determine the smallest value.", "which is the least among them?",
    "spot the minimum number.", "select the smallest number.", "pick out the smallest value.",
    "highlight the least number.", "which number is the lowest?", "locate the smallest number.",
    "distinguish the minimum value.", "choose the smallest number.", "point out the lowest number.",
    "decipher the smallest number.", "what is the least number?"
]

# Choose a random description template and question variation
description_template = random.choice(descriptions)
question_variation = random.choice(question_variations)

# Generate the description text including all the random numbers, marked with $$
text_numbers = " and ".join([f"$${num:.2f}&&" for num in random_floats])
description = f"{description_template}{text_numbers}, {question_variation}"

# Display the generated text and the smallest number
text = description
answer_numeric = answer_numeric

text, answer_numeric

import json

def generate_and_save_jsonl_alignment_numbertonumber(n, filename):
    with open(filename, 'w') as file:
        for _ in range(n):
            random_float = round(random.uniform(0, 100), 4)  # Generates a random float rounded to 4 decimal places
            data = {
                "input": str(random_float),  # Convert float to string
                "output": str(random_float)  # Convert float to string
            }
            json.dump(data, file)
            file.write('\n')  # Writes a new line for the next JSON object


def generate_and_save_jsonl_alignment_textnumber(n, filename):
    with open(filename, 'w') as file:
        for _ in range(n):
            random_float = round(random.uniform(0, 100), 4)  # Generates a random float rounded to 4 decimal places
            data = {
                "input": "$$"+str(random_float)+"&&",  # Convert float to string
                "output": str(random_float)  # Convert float to string
            }
            json.dump(data, file)
            file.write('\n')  # Writes a new line for the next JSON object


# Example usage
generate_and_save_jsonl_alignment_textnumber(5, 'alignmentTN.jsonl')

import numpy as np
import random
import json

def generate_questions_and_answers(m, filename):
    descriptions = [
        "Among the numbers ", "Considering the values ", "When comparing ", "Looking at the sequence ",
        "Given the numbers ", "Among these figures ", "From the list ", "Observing the numbers ",
        "In the collection ", "Considering the series ", "From this set ", "Taking into account ",
        "Given the array ", "Looking over ", "Reviewing the numbers ", "Within the group ",
        "From the array ", "Considering this selection ", "Examining the values ", "Analyzing the set ",
        "Among the chosen numbers ", "Selecting from ", "Highlighting the numbers ", "Given the selection ",
        "From these options ", "Among this collection ", "Considering these figures ", "From the numbers ",
        "Evaluating the set ", "Taking a look at "
    ]
    question_variations = [
        "what is the minimum number?", "can you identify the smallest value?", "which number is the least?",
        "find the lowest number.", "what's the smallest number among these?", "identify the minimum number.",
        "which of these is the smallest?", "determine the smallest value.", "which is the least among them?",
        "spot the minimum number.", "select the smallest number.", "pick out the smallest value.",
        "highlight the least number.", "which number is the lowest?", "locate the smallest number.",
        "distinguish the minimum value.", "choose the smallest number.", "point out the lowest number.",
        "decipher the smallest number.", "what is the least number?"
    ]

    with open(filename, 'w') as file:
        for _ in range(m):
            n = np.random.randint(1, 11)  # Generate a natural number n from 1 to 10
            random_floats = np.random.rand(n).astype(np.float32) * 100  # Generate n random float32 numbers
            answer_numeric = np.min(random_floats)  # Find the smallest number

            description_template = random.choice(descriptions)
            question_variation = random.choice(question_variations)

            text_numbers = " and ".join([f"$${num:.2f}&&" for num in random_floats])
            description = f"{description_template}{text_numbers}, {question_variation}"

            data = {
                "input": description,
                "output": f"{answer_numeric:.2f}"
            }

            json.dump(data, file)
            file.write('\n')

# Example usage to generate 100000 question-answer pairs

generate_questions_and_answers(100000, 'minimumalgo.jsonl')

generate_and_save_jsonl_alignment_textnumber(1000000, 'alignmentTN.jsonl')

generate_and_save_jsonl_alignment_numbertonumber(1000000, 'alignmentNN.jsonl')