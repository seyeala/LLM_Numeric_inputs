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

    def forward(self, inputs):
        if self.mixed_input:
            text_inputs, numeric_inputs = self._process_mixed_input(inputs['input_text'])
            numeric_embeds = self.input_projection(numeric_inputs.to(self.device))
            input_ids = self.tokenizer(text_inputs, return_tensors="pt")['input_ids'].to(self.device)
            text_embeds = self.model.transformer.wte(input_ids)
            combined_embeds = torch.cat([numeric_embeds.unsqueeze(0), text_embeds], dim=1).to(self.device)
            outputs = self.model(inputs_embeds=combined_embeds, return_dict=True)
        elif self.project_input and not self.mixed_input:
            embedded_input = self.input_projection(inputs.to(self.device))  # Ensure input tensor is on the correct device
            sequence_length = self.model.config.n_positions  # Use the maximum sequence length of the model
            inputs_embeds = embedded_input.unsqueeze(1).expand(-1, sequence_length, -1).to(self.device)
            position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(inputs.size(0), 1).to(self.device)
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids, return_dict=True, output_hidden_states=True)
        else:
            if isinstance(inputs, dict):
                # Ensuring that all tensors are moved to the correct device
                inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)

                if self.project_output and 'hidden_states' in outputs:
                    last_hidden_state = outputs.hidden_states[-1]
                    projected_output = self.output_projection(last_hidden_state[:, -1, :])
                    return projected_output

                return outputs.logits if hasattr(outputs, 'logits') else outpu
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move all input dictionary tensors to the device
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)

        if self.project_output and 'hidden_states' in outputs:
            last_hidden_state = outputs.hidden_states[-1].to(self.device)  # Move hidden states to the correct device
            projected_output = self.output_projection(last_hidden_state[:, -1, :].to(self.device))
            return projected_output

        return outputs.logits if hasattr(outputs, 'logits') else outputs

    def generate_text(self, input_text, **generate_kwargs):
        if not self.project_input and not self.project_output:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids, **generate_kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise NotImplementedError("Generate method is not implemented for projected input/output.")

    def _process_mixed_input(self, input_text):
        numeric_values = re.findall(r'\$\$(.*?)\&\&', input_text)
        numeric_values = [float(numeric) for numeric in numeric_values]
        processed_text = re.sub(r'\$\$.*?\&\&', '', input_text)
        numeric_inputs = torch.tensor(numeric_values, dtype=torch.float).view(-1, 1).to(self.device)
        return processed_text, numeric_inputs

    def configure_trainable_layers(self, train_input_projection=True, train_output_projection=True, train_transformer=True):
        """
        Configure which layers should be trainable and which should be frozen.
        """
        # Configure input projection layer
        if self.project_input and self.input_projection is not None:
            for param in self.input_projection.parameters():
                param.requires_grad = train_input_projection

        # Configure output projection layer
        if self.project_output and self.output_projection is not None:
            for param in self.output_projection.parameters():
                param.requires_grad = train_output_projection

        # Configure the main transformer model
        for param in self.model.parameters():
            param.requires_grad = train_transformer
