import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
from wrapperNM import NumericLMWrapper, print_cuda_memory, clear_cuda_memory
import time
import argparse
import yaml


def generate_data(batch_size, min_val, max_val, device):
    """Generates random data for inputs and targets within a specified range."""
    inputs = torch.rand(batch_size, 1).to(device) * (max_val - min_val) + min_val
    targets = torch.rand(batch_size, 1).to(device) * (max_val - min_val) + min_val
    return inputs, targets

clear_cuda_memory()
def alignment(llm, config, num_epochs, min_val, max_val, model_path, shl):
    llm.train()
    device = next(llm.parameters()).device
    optimizer = Adam(llm.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
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


    torch.save(llm.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with adjustable parameters.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--min_val", type=float, help="Minimum value for generated data.")
    parser.add_argument("--max_val", type=float, help="Maximum value for generated data.")
    parser.add_argument("--model_name", type=str, help="Model name for loading.")
    parser.add_argument("--shl", type=bool, help="Whether to use StepLR scheduler.")
    parser.add_argument("--model_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Override YAML settings with command-line arguments (if specified)
    config['num_epochs'] = args.num_epochs if args.num_epochs is not None else config.get('num_epochs', 2)
    config['min_val'] = args.min_val if args.min_val is not None else config.get('min_val', 0)
    config['max_val'] = args.max_val if args.max_val is not None else config.get('max_val', 100)
    config['model_name'] = args.model_name if args.model_name is not None else config.get('model_name', 'openai-community/gpt2-large')
    config['shl'] = args.shl if args.shl is not None else config.get('shl', False)
    config['model_path'] = args.model_path if args.model_path is not None else config.get('model_path', './chk/trained_numeric_lm.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    numeric_lm = NumericLMWrapper(config['model_name'], project_input=True, project_output=True, device=device)
    numeric_lm.configure_trainable_layers(train_input_projection=True, train_output_projection=True, train_transformer=False)

    alignment(numeric_lm, config, config['num_epochs'], config['min_val'], config['max_val'], config['model_path'], config['shl'])
