import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
import time
import argparse
import yaml
from wrapperNM import NumericLMWrapper, print_cuda_memory, clear_cuda_memory


def load_specific_weights(model, load_path):
    # Load the entire state dictionary from the file
    full_state_dict = torch.load(load_path)

    # Filter the state dict to only include weights for the projection layers and the language model
    filtered_state_dict = {name: param for name, param in full_state_dict.items()
                           if 'input_projection' in name or 'output_projection' in name or 'model' in name}

    # Load the filtered state dictionary into the model
    model.load_state_dict(filtered_state_dict, strict=False)  # Use strict=False to ignore non-matching keys

    return model


def generate_text_data(batch_size, min_val, max_val, device, tokenizer):
    """Generates text data for inputs."""
    inputs = torch.rand(batch_size, 1) * (max_val - min_val) + min_val
    text_inputs = ["$$" + str(number.item()) + "&&" for number in inputs]
    tensor_inputs = tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True).to(device)
    targets = torch.rand(batch_size, 1).to(device)  # Dummy targets for example
    return tensor_inputs, targets

def alignmenttext(llm, config, num_epochs, model_path_load, model_path_save, shl):
    device = next(llm.parameters()).device
    optimizer = Adam(filter(lambda p: p.requires_grad, llm.parameters()), lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
    scaler = GradScaler()

    # Load model state if exists
    if model_path_load:
        try:
            print(f"Attempting to load model from {model_path_load}")
            load_specific_weights(llm, model_path_load)
            print(f"Successfully loaded model from {model_path_load}")
        except FileNotFoundError:
            print(f"No model found at {model_path_load}, starting from scratch.")
        except RuntimeError as e:
            print(f"Failed to load model. Error: {e}")
            print(f"Check if the model architecture during saving matches the one during loading.")

    llm.train()

    for epoch in range(num_epochs):
        total_cpu_start = time.time()  # Start the timer for total CPU time this epoch
        cumulative_data_cpu_time = 0  # Initialize cumulative CPU data loading time
        epoch_loss_sum = 0  # Sum of losses for the epoch

        for i in range(config['num_batches']):
            batch_cpu_start = time.time()  # Start the timer for one batch
            batch_inputs, batch_targets = generate_text_data(config['batch_size'], config['min_val'], config['max_val'], device, llm.tokenizer)
            batch_cpu_end = time.time()  # End the timer for one batch
            cumulative_data_cpu_time += batch_cpu_end - batch_cpu_start  # Accumulate the CPU time

            optimizer.zero_grad()
            with autocast():
                outputs = llm(batch_inputs)  # Directly access the tensor output
                loss = nn.MSELoss()(outputs, batch_targets)
                epoch_loss_sum += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #print(f'Batch {i + 1}: Loss = {loss.item():.4f}, Batch CPU Time = {batch_cpu_end - batch_cpu_start:.2f} seconds')

        if shl and scheduler:
            scheduler.step()

        total_cpu_time = time.time() - total_cpu_start  # Total CPU time for the epoch
        average_loss = epoch_loss_sum / config['num_batches']  # Average loss for the epoch
        print(f'Epoch {epoch + 1}: Average Loss = {average_loss:.4f}, Total Compute Time = {total_cpu_time:.2f} seconds, Cumulative Data Loading CPU Time = {cumulative_data_cpu_time:.2f} seconds')
        print_cuda_memory()

    torch.save(llm.state_dict(), model_path_save)
    print(f"Saved trained model to {model_path_save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with text inputs.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--min_val", type=float, help="Minimum value for generated data.")
    parser.add_argument("--max_val", type=float, help="Maximum value for generated data.")
    parser.add_argument("--model_name", type=str, help="Model name for loading.")
    parser.add_argument("--shl", type=bool, help="Whether to use StepLR scheduler.")
    parser.add_argument("--model_path_load", type=str, help="Path to save the trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--model_path_save", help="Path to save the trained model.", default="./chk/atrained_numeric_lm_stage2.pth")

    # Parse arguments from command line
    args = parser.parse_args()

    # Load configuration from YAML file
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    # Apply default values first
    num_epochs = config.get('num_epochs', 2)  # Default value
    min_val = config.get('min_val', 0.0)  # Default value
    max_val = config.get('max_val', 100.0)  # Default value
    model_name = config.get('model_name', 'openai-community/gpt2-large')  # Default value
    shl = config.get('shl', False)  # Default value
    #model_path_load = config.get('model_path_load', './chk/trained_numeric_lm.pth')  # Default value
    #model_path_save = config.get('model_path_save', './chk/atrained_numeric_lm_stage2.pth')  # Default value

    # Override with command line arguments if provided
    if args.num_epochs is not None:
        num_epochs = args.num_epochs
    if args.min_val is not None:
        min_val = args.min_val
    if args.max_val is not None:
        max_val = args.max_val
    if args.model_name is not None:
        model_name = args.model_name
    if args.shl is not None:
        shl = args.shl
    if args.model_path_load is not None:
        model_path_load = args.model_path_load
    if args.model_path_save is not None:
        model_path_save = args.model_path_save

    # Training function or whatever you need to do
    print(f"Training with config: epochs={num_epochs}, min_val={min_val}, max_val={max_val}, model={model_name}, scheduler={shl}")
    print(f"Model will be saved to {model_path_save} and {model_path_save}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Example of instantiating the model for loading with project_input=True
    llm = NumericLMWrapper(config['model_name'], project_input=False, project_output=True, device=device)

    llm.configure_trainable_layers(train_input_projection=False, train_output_projection=True, train_transformer=False)
    llm.mixed_input=False

    # Ensure the tokenizer has a pad token set
    if llm.tokenizer.pad_token is None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token

    alignmenttext(llm, config, config['num_epochs'], args.model_path_load, args.model_path_save, config['shl'])
