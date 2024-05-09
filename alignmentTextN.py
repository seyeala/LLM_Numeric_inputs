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
from alignmentNN import alignment, clear_cuda_memory, print_cuda_memory



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
