import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
from wrapperNM import NumericLMWrapper
import time
import argparse
import yaml
import argparse
import yaml
from wrapperNM import NumericLMWrapper
from alignmentNN import clear_cuda_memory, print_cuda_memory


def alignmenttxt(llm, config, num_epochs, model_path, shl):
    llm.train()
    device = next(llm.parameters()).device
    optimizer = Adam(llm.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
    scaler = GradScaler()
    tokenizer = llm.tokenizer  # Assuming the tokenizer is part of the LLM wrapper

    for epoch in range(num_epochs):
        total_cpu_start = time.time()
        cumulative_data_cpu_time = 0
        epoch_loss_sum = 0

        for i in range(config['num_batches']):
            data_cpu_start = time.time()
            batch_inputs, batch_targets = generate_text_data(config['batch_size'], device, tokenizer)
            data_cpu_end = time.time()
            cumulative_data_cpu_time += data_cpu_end - data_cpu_start

            optimizer.zero_grad()
            with autocast():
                outputs = llm(**batch_inputs)
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

    torch.save(llm.state_dict(), './chk/slignment_txt2number')


def generate_text_data(batch_size, device, tokenizer):
    """Generates text data for inputs."""
    sentences = [
        "Hello, how are you today?",
        "The weather is great for a walk.",
        "I'm learning to use deep learning models.",
        "What is your favorite book?",
        "Deep learning transforms many industries."
    ]
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    targets = torch.rand(batch_size, 1).to(device)  # Dummy targets for example
    return inputs, targets

def main(config_path):
    parser = argparse.ArgumentParser(description="Train a model with adjustable parameters.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--model_name", type=str, help="Model name for loading.")
    parser.add_argument("--model_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--config", type=str, default=config_path, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Override YAML settings with command-line arguments (if specified)
    num_epochs = args.num_epochs if args.num_epochs is not None else config.get('num_epochs', 2)
    model_name = args.model_name if args.model_name is not None else config.get('model_name', 'openai-community/gpt2-large')
    model_path = args.model_path if args.model_path is not None else config.get('model_path', './trained_numeric_lm.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    numeric_lm = NumericLMWrapper(model_name, project_input=False, project_output=True, device=device)

    # Load the pre-trained model state if availablegf
    if args.model_path:
        model_state_dict = torch.load(args.model_path)
        numeric_lm.load_state_dict(model_state_dict)
        print("Loaded model weights from:", args.model_path)

    numeric_lm.configure_trainable_layers(train_input_projection=False, train_output_projection=True, train_transformer=False)

    alignment(numeric_lm, config['num_batches'], config['batch_size'], config['lr'], num_epochs, config['min_val'], config['max_val'], model_path, config['shl'])

if __name__ == "__main__":
    main("config.yml")
