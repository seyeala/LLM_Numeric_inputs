import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from wrapperNM import NumericLMWrapper
import time

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

def alignment(llm, num_batches, batch_size, lr, num_epochs, min_val, max_val):
    llm.train()
    device = next(llm.parameters()).device
    optimizer = Adam(llm.parameters(), lr=lr)
    scaler = GradScaler()

    cumulative_cpu_time = 0
    cumulative_gpu_time = 0

    for epoch in range(num_epochs):
        # Start the CPU timer
        cpu_start = time.time()

        # Initialize GPU timers
        gpu_start_event = torch.cuda.Event(enable_timing=True)
        gpu_end_event = torch.cuda.Event(enable_timing=True)

        # Start the GPU timer
        gpu_start_event.record()

        for i in range(num_batches):
            batch_inputs, batch_targets = generate_data(batch_size, min_val, max_val, device)

            optimizer.zero_grad()
            with autocast():
                outputs = llm(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if i % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

        # End the CPU timer
        cpu_end = time.time()
        cumulative_cpu_time += cpu_end - cpu_start

        # End the GPU timer
        gpu_end_event.record()
        torch.cuda.synchronize()  # Wait for the GPU event to complete
        cumulative_gpu_time += gpu_start_event.elapsed_time(gpu_end_event) / 1000  # Convert ms to seconds

        print(f'End of Epoch {epoch+1}')
        print(f'Cumulative CPU Time: {cumulative_cpu_time:.2f} seconds')
        print(f'Cumulative GPU Time: {cumulative_gpu_time:.2f} seconds')
        print_cuda_memory()
        clear_cuda_memory()

    torch.save(llm.state_dict(), 'trained_numeric_lm.pth')

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "openai-community/gpt2-large"
numeric_lm = NumericLMWrapper(model_name, project_input=True, project_output=True, device=device)
numeric_lm.configure_trainable_layers(train_input_projection=True, train_output_projection=True, train_transformer=False)

alignment(numeric_lm, num_batches=100, batch_size=5, lr=0.001, num_epochs=10, min_val=0, max_val=100)
