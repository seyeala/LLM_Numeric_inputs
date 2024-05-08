import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class NumericLMWrapper(nn.Module):
    def __init__(self, model_name, project_input=False, project_output=False, mixed_input=False):
        super(NumericLMWrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.project_input = project_input
        self.project_output = project_output
        self.mixed_input = mixed_input
        embedding_dim = self.model.config.hidden_size

        if self.project_input:
            self.input_projection = nn.Linear(1, embedding_dim)

        if self.project_output:
            self.output_projection = nn.Linear(embedding_dim, 1)

    def forward(self, inputs):
        if self.project_input:
            if isinstance(inputs, dict) and 'input_ids' in inputs:
                # Extract tensor from dictionary
                inputs = inputs['input_ids']

            # Assuming inputs is a tensor for numeric input
            embedded_input = self.input_projection(inputs)  # Shape: [batch_size, embedding_dim]
            sequence_length = self.model.config.n_positions  # Use the maximum sequence length of the model
            inputs_embeds = embedded_input.unsqueeze(1).expand(-1, sequence_length, -1)
            position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(inputs.size(0), 1).to(inputs.device)
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids, return_dict=True, output_hidden_states=True)
        else:
            # Assume inputs is a dictionary with text input
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)

        if self.project_output and 'hidden_states' in outputs:
            last_hidden_state = outputs.hidden_states[-1]  # Use the last hidden state from the outputs
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
            return projected_output

        # Return logits or token ids if not projecting output
        return outputs.logits if hasattr(outputs, 'logits') else outputs



# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=False)

# Example of text input and getting output
inputs = {"input_ids": numeric_lm.tokenizer.encode("Who are you.", return_tensors="pt")}
output = numeric_lm(inputs)  # Passing dictionary when project_input is False
print(inputs,inputs,output)



probabilities = torch.nn.functional.softmax(output, dim=-1)

# Get the most probable next token indices
_, predicted_indices = torch.max(probabilities, dim=-1)

# Decode the token indices to text
decoded_text = numeric_lm.tokenizer.decode(predicted_indices.tolist()[0])  # Assuming single batch

print('ff', decoded_text)

# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=True)

# Example of text input and getting output
inputs = {"input_ids": numeric_lm.tokenizer.encode("Hello how are you?.", return_tensors="pt")}
output = numeric_lm(inputs)  # Passing dictionary when project_input is False
print('FT', output)
# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True)

# Example of numeric input and getting numeric output
input_numeric = torch.tensor([[0.5]])  # Example numeric batch input
output = numeric_lm(input_numeric)  # Ensured input is already 2D
print('TT', output)



model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=False)
