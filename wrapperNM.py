import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def clear_cuda_memory():
    """Clears unused memory from CUDA memory cache."""
    torch.cuda.empty_cache()
    print("Cleared CUDA memory cache.")



def print_cuda_memory():
    """Prints the current and maximum memory used on CUDA."""
    print(f'Current memory allocated: {torch.cuda.memory_allocated() / 1e6} MB')
    print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6} MB')
    torch.cuda.reset_peak_memory_stats()

class NumericLMWrapper(nn.Module):
    def __init__(self, model_name, project_input=False, project_output=False, mixed_input=False, device='cpu'):
        super(NumericLMWrapper, self).__init__()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.project_input = project_input
        self.project_output = project_output
        self.mixed_input = mixed_input
        embedding_dim = self.model.config.hidden_size

        if self.project_input:
            self.input_projection = nn.Linear(1, embedding_dim).to(self.device)

        if self.project_output:
            self.output_projection = nn.Linear(embedding_dim, 1).to(self.device)

    def forward(self, inputs, **kwargs):
        # Check if inputs is a dictionary with 'input_ids', if not, assume numeric or mixed input
        if isinstance(inputs, dict) and 'input_ids' in inputs:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)
        elif self.mixed_input and 'input_text' in kwargs:
            text_inputs, numeric_inputs = self._process_mixed_input(kwargs['input_text'])
            numeric_embeds = self.input_projection(numeric_inputs)
            input_ids = self.tokenizer(text_inputs, return_tensors="pt")['input_ids'].to(self.device)
            text_embeds = self.model.transformer.wte(input_ids)
            combined_embeds = torch.cat([numeric_embeds.unsqueeze(0), text_embeds], dim=1)
            outputs = self.model(inputs_embeds=combined_embeds, return_dict=True)
        else:
            # Handling numeric inputs for projection
            embedded_input = self.input_projection(inputs.to(self.device))
            sequence_length = self.model.config.n_positions
            inputs_embeds = embedded_input.unsqueeze(1).expand(-1, sequence_length, -1)
            position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(inputs.size(0), 1).to(self.device)
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids, return_dict=True)

        if self.project_output and 'hidden_states' in outputs:
            last_hidden_state = outputs.hidden_states[-1]
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs

    def _process_mixed_input(self, input_text):
        numeric_values = re.findall(r'\$\$(.*?)\&\&', input_text)
        numeric_values = [float(numeric) for numeric in numeric_values]
        processed_text = re.sub(r'\$\$.*?\&\&', '', input_text)
        numeric_inputs = torch.tensor(numeric_values, dtype=torch.float).view(-1, 1).to(self.device)
        return processed_text, numeric_inputs

    def configure_trainable_layers(self, train_input_projection=True, train_output_projection=True, train_transformer=True):
        # Configures which layers are trainable
        if self.project_input and self.input_projection is not None:
            for param in self.input_projection.parameters():
                param.requires_grad = train_input_projection
        if self.project_output and self.output_projection is not None:
            for param in self.output_projection.parameters():
                param.requires_grad = train_output_projection
        for param in self.model.parameters():
            param.requires_grad = train_transformer
