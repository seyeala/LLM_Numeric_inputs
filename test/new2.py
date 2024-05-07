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

        if self.project_input:
            self.input_projection = nn.Linear(1, embedding_dim)

        if self.project_output:
            self.output_projection = nn.Linear(embedding_dim, 1)

    def forward(self, inputs):
        if self.project_input:
            # Assume inputs is a batch of single numeric values
            embedded_input = self.input_projection(inputs.unsqueeze(-1))  # Adding missing batch dimension
            outputs = self.model(inputs_embeds=embedded_input.repeat(1, self.model.config.n_positions, 1))
        else:
            outputs = self.model(**inputs)

        if self.project_output:
            # Assuming you are looking to output the projection of the last token's embedding
            logits = outputs.logits  # Ensure logits are being generated
            last_hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else logits
            projected_output = self.output_projection(last_hidden_state[:, -1, :])  # Project last token
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs

# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True)

# Example of numeric input and getting numeric output
input_numeric = torch.tensor([0.5, 1.5, 2.5])  # Example numeric batch input
output = numeric_lm(input_numeric)
print(output)
