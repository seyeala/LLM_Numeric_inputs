import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from wrapperNM import NumericLMWrapper

def clear_cuda_memory():
    """Clears unused memory from CUDA memory cache."""
    torch.cuda.empty_cache()
    print("Cleared CUDA memory cache.")

def print_cuda_memory():
    """Prints the current and maximum memory used on CUDA."""
    print(f'Current memory allocated: {torch.cuda.memory_allocated() / 1e6} MB')
    print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6} MB')
    torch.cuda.reset_peak_memory_stats()  # Reset the peak memory metric after printing

def generate_data(batch_size, min_val, max_val, device):
    """Generates random data for inputs and targets within a specified range."""
    inputs = torch.rand(batch_size, 1).to(device) * (max_val - min_val) + min_val
    targets = torch.rand(batch_size, 1).to(device) * (max_val - min_val) + min_val
    return inputs, targets

def alignment(llm, num_batches, batch_size, lr, num_epochs, min_val, max_val):
    """Trains the LLM with specified hyperparameters, generating data on the fly."""
    llm.train()  # Set the model to training mode
    device = next(llm.parameters()).device  # Get the device of the model

    # Setup optimizer and scaler for mixed precision
    optimizer = Adam(llm.parameters(), lr=lr)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch_inputs, batch_targets = generate_data(batch_size, min_val, max_val, device)

            optimizer.zero_grad()
            with autocast():  # Enable mixed precision
                outputs = llm(batch_inputs)  # Forward pass
                loss = nn.MSELoss()(outputs, batch_targets)

            scaler.scale(loss).backward()  # Scale the loss to prevent underflow
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the scale for next iteration
            optimizer.zero_grad()

            if i % 10 == 0:  # Print loss every 10 batches
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')
        print_cuda_memory()


    # Save the trained model state
    torch.save(llm.state_dict(), './chk/alignment_number2number/trained_numeric_lm.pth')

# Example usage
clear_cuda_memory()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "openai-community/gpt2-large"
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True, device=device)
numeric_lm.configure_trainable_layers(train_input_projection=True, train_output_projection=True, train_transformer=False)

# Call the alignment function with min and max values for the random numbers
alignment(numeric_lm, num_batches=100, batch_size=5, lr=0.001, num_epochs=10, min_val=0, max_val=1)
