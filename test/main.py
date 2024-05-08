import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from torch.cuda.amp import autocast, GradScaler
from wrapperNM import NumericLMWrapper
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# Assuming NumericLMWrapper is imported and configured as above
model_name = "openai-community/gpt2-large"  # substitute with the actual model you are using
# Assuming NumericLMWrapper is imported and configured
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True, device='cuda')
numeric_lm.train()  # Set the model to training mode

# Setup data
inputs = torch.rand(100, 1).cuda()  # Example numeric inputs
targets = torch.rand(100, 1).cuda()  # Example targets

dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Setup optimizer and scaler for mixed precision
optimizer = Adam(numeric_lm.parameters(), lr=0.001)
scaler = GradScaler()

num_epochs = 10
accumulation_steps = 4  # Number of batches to accumulate gradients over

for epoch in range(num_epochs):
    for i, (batch_inputs, batch_targets) in enumerate(dataloader):
        optimizer.zero_grad()

        with autocast():  # Enable mixed precision
            outputs = numeric_lm(batch_inputs)  # Forward pass
            loss = torch.nn.MSELoss()(outputs, batch_targets)

        scaler.scale(loss).backward()  # Scale the loss to prevent underflow

        # Perform parameter update every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the scale for next iteration
            optimizer.zero_grad()

        print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

torch.save(numeric_lm.state_dict(), 'trained_numeric_lm.pth')
