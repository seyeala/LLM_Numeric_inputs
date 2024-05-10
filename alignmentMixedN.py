import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
import time
import argparse
import yaml
from wrapperNM import NumericLMWrapper, print_cuda_memory, clear_cuda_memory,load_specific_weights

def _process_mixed_input(self, text_inputs):
    numeric_values = []
    processed_texts = []

    for text_input in text_inputs:
        found_numbers = re.findall(r'\$\$(.*?)\&\&', text_input)
        found_numbers = [float(num) for num in found_numbers]
        numeric_values.extend(founds_numbers)

        # Remove the numeric placeholders after extracting values
        processed_text = re.sub(r'\$\$.*?\&\&', '', text_input)
        processed_texts.append(processed_text)

    numeric_inputs = torch.tensor(numeric_values, dtype=torch.float).view(-1, 1).to(self.device)

    # Join the processed texts if necessary or handle them as a batch if your model requires that
    processed_text = ' '.join(processed_texts)

    return processed_text, numeric_inputs



def alignmentmixed(llm, config, num_epochs, model_path_load, model_path_save, shl):
    device = next(llm.parameters()).device
    optimizer = Adam(filter(lambda p: p.requires_grad, llm.parameters()), lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
    scaler = GradScaler()

    # Load model state if exists
    if model_path_load:
        try:
            llm.load_state_dict(torch.load(model_path_load))
            print("Successfully loaded model from:", model_path_load)
        except FileNotFoundError:
            print("No model found at:", model_path_load)

    llm.train()

    for epoch in range(num_epochs):
        for i in range(config['num_batches']):
            batch_inputs, batch_targets = generate_text_data(config['batch_size'], config['min_val'], config['max_val'], device, llm.tokenizer)
            optimizer.zero_grad()
            with autocast():
                # Directly pass the entire `batch_inputs` dictionary when mixed input is enabled
                outputs = llm(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if shl and scheduler:
            scheduler.step()

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
    llm = NumericLMWrapper(config['model_name'], project_input=True, project_output=True, mixed_input=True, device=device)

    llm.configure_trainable_layers(train_input_projection=False, train_output_projection=True, train_transformer=False)

    # Ensure the tokenizer has a pad token set
    if llm.tokenizer.pad_token is None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token

    alignmentmixed(llm, config, config['num_epochs'], args.model_path_load, args.model_path_save, config['shl'])
