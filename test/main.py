import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

from wrapperNM import NumericLMWrapper
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# Assuming NumericLMWrapper is imported and configured as above
model_name = "openai-community/gpt2-large"  # substitute with the actual model you are using

numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True, device='cuda')
numeric_lm.train()  # Set the model to training mode

# Example dummy data (replace with your actual data)
inputs = torch.rand(100, 1).cuda()  # 100 random numbers as input
targets = torch.rand(100, 1).cuda()  # 100 random numbers as target output

# Create a simple dataset and dataloader
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = Adam(numeric_lm.parameters(), lr=0.001)
