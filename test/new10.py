import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class NumericLMWrapper(nn.Module):
    def __init__(self, model_name, project_input=False, project_output=False, mixed_input=False, device='cpu'):
        super(NumericLMWrapper, self).__init__()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.project_input = project_input
        self.project_output = project_output
        self.mixed_input = mixed_input
        self.embedding_dim = self.model.config.hidden_size

        if self.project_input:
            self.input_projection = nn.Linear(1, self.embedding_dim).to(self.device)

        if self.project_output:
            self.output_projection = nn.Linear(self.embedding_dim, 1).to(self.device)

    def forward(self, inputs):
        if self.mixed_input:
            text_inputs, numeric_inputs = self._process_mixed_input(inputs['input_text'])
            numeric_embeds = self.input_projection(numeric_inputs.to(self.device))
            input_ids = self.tokenizer(text_inputs, return_tensors="pt").to(self.device)['input_ids']
            text_embeds = self.model.transformer.wte(input_ids)

            combined_embeds = torch.cat([numeric_embeds.unsqueeze(0), text_embeds], dim=1)
            outputs = self.model(inputs_embeds=combined_embeds, return_dict=True)

        elif self.project_input:
            embedded_input = self.input_projection(inputs.to(self.device))
            sequence_length = self.model.config.n_positions
            inputs_embeds = embedded_input.unsqueeze(1).expand(-1, sequence_length, -1)
            position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(inputs.size(0), 1).to(self.device)
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids, return_dict=True)

        else:
            # Assume inputs is a dictionary with text input already on the correct device
            outputs = self.model(**inputs, return_dict=True)

        if self.project_output and 'hidden_states' in outputs:
            last_hidden_state = outputs.hidden_states[-1]
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs

    def _process_mixed_input(self, input_text):
        numeric_values = re.findall(r'\$\$(.*?)\&\&', input_text)
        numeric_values = [float(numeric) for numeric in numeric_values]
        processed_text = re.sub(r'\$\$.*?\&\&', '', input_text)
        numeric_inputs = torch.tensor(numeric_values, dtype=torch.float).view(-1, 1)
        return processed_text, numeric_inputs

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "EleutherAI/gpt-j-6b"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True, mixed_input=True, device=device)

# Mixed input example
input_text = "Hello $$100.5&& world $$200.1&&!"
inputs = {"input_text": input_text}
output = numeric_lm(inputs)
print(input_text, output)

# Example usage
model_name = "EleutherAI/gpt-j-6b" # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=False, device=device)

# Example of text input and getting output
inputs = {"input_ids": numeric_lm.tokenizer.encode("Hello, world!", return_tensors="pt")}
output = numeric_lm(inputs)  # Passing dictionary when project_input is False
print(output)



# Example usage
model_name = "EleutherAI/gpt-j-6b"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=False, device=device)

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
model_name = "EleutherAI/gpt-j-6b" # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=True, device=device)

# Example of text input and getting output
inputs = {"input_ids": numeric_lm.tokenizer.encode("Hello how are you?.", return_tensors="pt")}
output = numeric_lm(inputs)  # Passing dictionary when project_input is False
print('FT', output)
# Example usage
model_name = "EleutherAI/gpt-j-6b"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True, device=device)

# Example of numeric input and getting numeric output
input_numeric = torch.tensor([[0.5]])  # Example numeric batch input
output = numeric_lm(input_numeric)  # Ensured input is already 2D
print('TT', output)



model_name = "EleutherAI/gpt-j-6b"  # substitute with the actual model you are using
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=False)


model_name = "EleutherAI/gpt-j-6b"
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=False, mixed_input=True)

# Mixed input example
input_text = "Hello $$100.5&& world $$200.1&&!"
inputs = {"input_text": input_text}
output = numeric_lm(inputs)
print(input_text,output)


probabilities = torch.nn.functional.softmax(output, dim=-1)

# Get the most probable next token indices
_, predicted_indices = torch.max(probabilities, dim=-1)

# Decode the token indices to text
decoded_text = numeric_lm.tokenizer.decode(predicted_indices.tolist()[0])  # Assuming single batch

print( decoded_text)
