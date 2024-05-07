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
            # Assuming inputs has a shape [batch_size, 1] where 1 is the numeric input per example
            embedded_input = self.input_projection(inputs)  # Shape: [batch_size, embedding_dim]

            # Expand input embeddings across the sequence length
            sequence_length = self.model.config.n_positions  # Use the maximum sequence length of the model
            inputs_embeds = embedded_input.unsqueeze(1).repeat(1, sequence_length, 1)

            # Generate position IDs for each position in the sequence
            position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(inputs.size(0), 1).to(inputs.device)

            # Feed into the model
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        else:
            # Standard model input handling
            outputs = self.model(**inputs)

        if self.project_output:
            # Extracting the last token's hidden state to project to a numeric output
            last_hidden_state = outputs.last_hidden_state  # Assuming this exists, check your model's output
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs

# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True)

# Example of numeric input and getting numeric output
input_numeric = torch.tensor([[0.5], [1.5], [2.5]])  # Example numeric batch input
output = numeric_lm(input_numeric.unsqueeze(-1))  # Ensure input is 2D
print(output)
