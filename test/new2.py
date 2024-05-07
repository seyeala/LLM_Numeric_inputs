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
        if self.mixed_input:
            text_inputs, numeric_inputs = self._process_mixed_input(inputs['input_text'])
            numeric_embeds = self.input_projection(numeric_inputs)  # Shape: [num_numeric_values, embedding_dim]

            # Tokenize the text inputs and get embeddings
            input_ids = self.tokenizer(text_inputs, return_tensors="pt")['input_ids'].to(numeric_inputs.device)
            text_embeds = self.model.transformer.wte(input_ids)  # Shape: [batch_size, seq_length, embedding_dim]

            # We need to adjust how we handle the concatenation of numeric and text embeddings
            # Option: Repeat numeric embeddings to match sequence length of text embeddings if necessary
            if numeric_embeds.size(0) < text_embeds.size(1):
                # Repeating the numeric embeddings to cover the whole sequence length
                repeats = text_embeds.size(1) // numeric_embeds.size(0)
                numeric_embeds = numeric_embeds.repeat(repeats, 1)[:text_embeds.size(1), :]

            # Combine embeddings
            combined_embeds = torch.cat([numeric_embeds.unsqueeze(0), text_embeds], dim=1)

            outputs = self.model(inputs_embeds=combined_embeds, return_dict=True)
        else:
            outputs = self.model(**inputs, return_dict=True)

        if self.project_output and 'hidden_states' in outputs:
            last_hidden_state = outputs.hidden_states[-1]
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs


    def _process_mixed_input(self, input_text):
        # Extract numeric values between $$ and &&
        numeric_values = re.findall(r'\$\$(.*?)\&\&', input_text)
        numeric_values = [float(numeric) for numeric in numeric_values]

        # Replace numeric values in text with a placeholder or remove
        processed_text = re.sub(r'\$\$.*?\&\&', '', input_text)

        # Convert numeric values to tensor and project
        numeric_inputs = torch.tensor(numeric_values, dtype=torch.float).view(-1, 1)

        return processed_text, numeric_inputs

# Example usage
model_name = "gpt2"
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=False, mixed_input=True)

# Mixed input example
input_text = "Hello $$100.5&& world $$200.1&&!"
inputs = {"input_text": input_text}
output = numeric_lm(inputs)
print(input_text,output)

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

print(decoded_text)

# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=True)

# Example of text input and getting output
inputs = {"input_ids": numeric_lm.tokenizer.encode("Hello how are you?.", return_tensors="pt")}
output = numeric_lm(inputs)  # Passing dictionary when project_input is False
print(output)
# Example usage
model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True)

# Example of numeric input and getting numeric output
input_numeric = torch.tensor([[0.5]])  # Example numeric batch input
output = numeric_lm(input_numeric)  # Ensured input is already 2D
print(output)



model_name = "gpt2"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=Tr, project_output=False)

# Example of text input and getting output
inputs = {"input_ids": numeric_lm.tokenizer.encode("Hello, world!", return_tensors="pt")}
output = numeric_lm(inputs)  # Passing dictionary when project_input is False
print(output)
