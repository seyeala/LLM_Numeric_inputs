import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
from wrapperNM import NumericLMWrapper
import time
import argparse
import yaml

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

def alignment(llm, config, num_epochs, min_val, max_val, model_path, shl):
    llm.train()
    device = next(llm.parameters()).device
    optimizer = Adam(llm.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_cpu_start = time.time()
        cumulative_data_cpu_time = 0
        epoch_loss_sum = 0

        for i in range(config['num_batches']):
            data_cpu_start = time.time()
            batch_inputs, batch_targets = generate_data(config['batch_size'], min_val, max_val, device)
            data_cpu_end = time.time()
            cumulative_data_cpu_time += data_cpu_end - data_cpu_start

            optimizer.zero_grad()
            with autocast():
                outputs = llm(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)
                epoch_loss_sum += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_cpu_time = time.time() - total_cpu_start
        average_loss = epoch_loss_sum / config['num_batches']

        print(f'End of Epoch {epoch+1}')
        print(f'Total Compute Time for Epoch: {total_cpu_time:.2f} seconds')
        print(f'Cumulative Data Loading CPU Time: {cumulative_data_cpu_time:.2f} seconds')
        print(f'Average Loss for Epoch: {average_loss:.4f}')
        if shl and scheduler:
            scheduler.step()
        print_cuda_memory()
        clear_cuda_memory()

    torch.save(llm.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with adjustable parameters.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs to train.")
    parser.add_argument("--min_val", type=float, default=0, help="Minimum value for generated data.")
    parser.add_argument("--max_val", type=float, default=100, help="Maximum value for generated data.")
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2-large", help="Model name for loading.")
    parser.add_argument("--shl", type=bool, default=False, help="Whether to use StepLR scheduler.")
    parser.add_argument("--model_path", type=str, default='trained_numeric_lm.pth', help="Path to save the trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    numeric_lm = NumericLMWrapper(args.model_name, project_input=True, project_output=True, device=device)
    numeric_lm.configure_trainable_layers(train_input_projection=True, train_output_projection=True, train_transformer=False)

    alignment(numeric_lm, config, args.num_epochs, args.min_val, args.max_val, args.model_path, args.shl)
