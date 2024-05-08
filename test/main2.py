import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from wrapperNM import NumericLMWrapper
import time
from torch.optim.lr_scheduler import StepLR


def generate_data(batch_size, min_val, max_val, device):
    """Generates random data for inputs and targets within a specified range."""
    inputs = torch.rand(batch_size, 1).to(device) * (max_val - min_val) + min_val
    targets = torch.rand(batch_size, 1).to(device) * (max_val - min_val) + min_val
    return inputs, targets

def print_cuda_memory():
    """Prints the current and maximum memory used on CUDA."""
    print(f'Current memory allocated: {torch.cuda.memory_allocated() / 1e6} MB')
    print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6} MB')
    torch.cuda.reset_peak_memory_stats()

def clear_cuda_memory():
    """Clears unused memory from CUDA memory cache."""
    torch.cuda.empty_cache()
    print("Cleared CUDA memory cache.")

def alignment(llm, num_batches, batch_size, lr, num_epochs, min_val, max_val, shl=False):
    llm.train()
    device = next(llm.parameters()).device
    optimizer = Adam(llm.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_cpu_start = time.time()  # Start total CPU timer for the epoch
        cumulative_data_cpu_time = 0
        epoch_loss_sum = 0  # Initialize the loss sum for the epoch

        for i in range(num_batches):
            # Measure data loading time separately
            data_cpu_start = time.time()
            batch_inputs, batch_targets = generate_data(batch_size, min_val, max_val, device)
            data_cpu_end = time.time()
            cumulative_data_cpu_time += data_cpu_end - data_cpu_start

            optimizer.zero_grad()
            with autocast():
                outputs = llm(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)
                epoch_loss_sum += loss.item()  # Add current batch loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if i % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

        total_cpu_end = time.time()  # End total CPU timer for the epoch
        total_cpu_time = total_cpu_end - total_cpu_start
        average_loss = epoch_loss_sum / num_batches  # Calculate average loss for the epoch

        print(f'End of Epoch {epoch+1}')
        print(f'Total Compute Time for Epoch: {total_cpu_time:.2f} seconds')
        print(f'Cumulative Data Loading CPU Time: {cumulative_data_cpu_time:.2f} seconds')
        print(f'Average Loss for Epoch: {average_loss:.4f}')  # Print the average loss
        if shl==True:
            scheduler.step()
        print_cuda_memory()



    torch.save(llm.state_dict(), 'trained_numeric_lm.pth')

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "openai-community/gpt2-large"
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True, device=device)
numeric_lm.configure_trainable_layers(train_input_projection=True, train_output_projection=True, train_transformer=False)

alignment(numeric_lm, num_batches=10, batch_size=8, lr=0.001, num_epochs=10, min_val=0, max_val=100, shl=True)
