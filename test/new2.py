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

            # Define the number of positions for which you want to expand your input embeddings
            sequence_length = 1  # Adjust this as necessary for your application

            # Generating position IDs for each position in the sequence
            position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(inputs.shape[0], 1).to(inputs.device)

            # Repeating the embeddings across the sequence length if necessary
            inputs_embeds = embedded_input.unsqueeze(1).repeat(1, sequence_length, 1)

            # Feed into the model
            model_inputs = {
                'inputs_embeds': inputs_embeds,
                'position_ids': position_ids,
                'return_dict': True,
                'output_hidden_states': True
            }
            outputs = self.model(**model_inputs)
        else:
            # Here, inputs should be a dictionary with keys that the model expects
            if isinstance(inputs, torch.Tensor):
                # If inputs is just a tensor, wrap it in a dictionary assuming it's input_ids
                inputs = {'input_ids': inputs}
            outputs = self.model(**inputs)

        if self.project_output:
            # Extracting the last token's hidden state to project to a numeric output
            last_hidden_state = outputs.last_hidden_state  # Assuming this exists, check your model's output
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs


        if self.project_output:
            last_hidden_state = outputs.hidden_states[-1]  # Use the last hidden state from the outputs
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs

# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True)

# Example of numeric input and getting numeric output
input_numeric = torch.tensor([[0.5]])  # Example numeric batch input, ensure input is 2D
output = numeric_lm(input_numeric)
print("Projected Output:", output)



numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=True)

# Prepare a text input, tokenize it, and get numeric output
text = "Example text input."
input_ids = numeric_lm.tokenizer.encode(text, return_tensors="pt")  # Tokenize text to input_ids
output = numeric_lm(input_ids)  # Now expecting token IDs, not raw numeric values
print("Projected Output:", output)
