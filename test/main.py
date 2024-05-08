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


# Create a simple dataset and dataloader
dataset = TensorDataset(inputs, inputs)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = Adam(numeric_lm.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = numeric_lm(batch_inputs)  # Forward pass
        loss = criterion(outputs, batch_targets)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the model
torch.save(numeric_lm.state_dict(), 'trained_numeric_lm.pth')
