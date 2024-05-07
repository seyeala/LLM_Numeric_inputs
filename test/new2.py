import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class NumericLMWrapper(nn.Module):
    def __init__(self, model_name, project_input=False, project_output=False):
        super(NumericLMWrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.project_input = project_input
        self.project_output = project_output
        embedding_dim = self.model.config.hidden_size

        # Input projection layer: projects a single numeric value to the embedding space
        if self.project_input:
            self.input_projection = nn.Linear(1, embedding_dim)

        # Output projection layer: projects the output embeddings to a single numeric value
        if self.project_output:
            self.output_projection = nn.Linear(embedding_dim, 1)

    def forward(self, inputs):
        if self.project_input:
            # Assume inputs is a batch of single numeric values
            embedded_input = self.input_projection(inputs.unsqueeze(-1))  # Add dimension for batch handling
            outputs = self.model(inputs_embeds=embedded_input)
        else:
            outputs = self.model(**inputs)

        if self.project_output:
            # Use the last hidden state for projection
            logits = outputs.logits
            last_hidden_state = logits[:, -1, :]  # Use the last token representation
            projected_output = self.output_projection(last_hidden_state)
            return projected_output

        return outputs

# Usage Example
model_name = "gpt2"  # or any other model from Hugging Face
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True)

# Example of using the model with numeric input and output projection
input_numeric = torch.tensor([0.5, 1.5, 2.5])  # Example numeric inputs
output = numeric_lm(input_numeric)
print(output)
