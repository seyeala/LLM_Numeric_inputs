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
        embedding_dim = self.model.config.hidden_size

        if self.project_input:
            self.input_projection = nn.Linear(1, embedding_dim).to(self.device)

        if self.project_output:
            self.output_projection = nn.Linear(embedding_dim, 1).to(self.device)

    def forward(self, **inputs):
        if self.mixed_input:
            # Assuming inputs['input_text'] contains the mixed text
            text_inputs, numeric_inputs = self._process_mixed_input(inputs['input_text'])
            numeric_embeds = self.input_projection(numeric_inputs.to(self.device))
            input_ids = self.tokenizer(text_inputs, return_tensors="pt")['input_ids'].to(self.device)
            text_embeds = self.model.transformer.wte(input_ids)
            combined_embeds = torch.cat([numeric_embeds.unsqueeze(0), text_embeds], dim=1).to(self.device)
            outputs = self.model(inputs_embeds=combined_embeds, return_dict=True)
        elif self.project_input and not self.mixed_input:
            # If inputs is expected to be a numeric tensor directly
            numeric_inputs = inputs['numeric_inputs'].to(self.device)  # Ensure this is named appropriately depending on how inputs are passed
            embedded_input = self.input_projection(numeric_inputs)
            sequence_length = self.model.config.n_positions
            inputs_embeds = embedded_input.unsqueeze(1).expand(-1, sequence_length, -1).to(self.device)
            position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(numeric_inputs.size(0), 1).to(self.device)
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids, return_dict=True, output_hidden_states=True)
        else:
            # Normalize inputs for tensor input handling for standard text input scenarios
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)

        if self.project_output and 'hidden_states' in outputs:
            last_hidden_state = outputs.hidden_states[-1]
            projected_output = self.output_projection(last_hidden_state[:, -1, :])
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
