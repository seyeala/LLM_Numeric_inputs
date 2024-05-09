import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
import time
import argparse
import yaml
from wrapperNM import NumericLMWrapper

def generate_text_data(batch_size, min_val, max_val, device, tokenizer):
    """Generates text data for inputs."""
    inputs = torch.rand(batch_size, 1) * (max_val - min_val) + min_val
    text_inputs = ["$$" + str(number.item()) + "&&" for number in inputs]
    tensor_inputs = tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True).to(device)
    targets = torch.rand(batch_size, 1).to(device)  # Dummy targets for example
    return tensor_inputs, targets

def alignment(llm, config, num_epochs, load_model_path, save_model_path, shl):
    device = next(llm.parameters()).device
    optimizer = Adam(filter(lambda p: p.requires_grad, llm.parameters()), lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
    scaler = GradScaler()

    # Load model state if exists
    if load_model_path:
        try:
            model_state_dict = torch.load(load_model_path)
            llm.load_state_dict(model_state_dict)
            print(f"Loaded model from {load_model_path}")
        except FileNotFoundError:
            print(f"No model found at {load_model_path}, starting from scratch.")

    llm.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(config['num_batches']):
            batch_inputs, batch_targets = generate_text_data(config['batch_size'], config['min_val'], config['max_val'], device, llm.tokenizer)

            optimizer.zero_grad()
            with autocast():
                output_dict = llm(**batch_inputs)
                outputs = output_dict.get('logits', output_dict)  # Modify based on your model's output
                loss = nn.MSELoss()(outputs, batch_targets)
                total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if shl and scheduler:
            scheduler.step()

        average_loss = total_loss / config['num_batches']
        print(f'Epoch {epoch + 1}/{num_epochs} - Average Loss: {average_loss:.4f}')
        print_cuda_memory()
        clear_cuda_memory()

    torch.save(llm.state_dict(), save_model_path)
    print(f"Saved trained model to {save_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with text inputs.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--min_val", type=float, help="Minimum value for generated data.")
    parser.add_argument("--max_val", type=float, help="Maximum value for generated data.")
    parser.add_argument("--model_name", type=str, help="Model name for loading.")
    parser.add_argument("--shl", type=bool, help="Whether to use StepLR scheduler.")
    parser.add_argument("--model_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--savestage2", help="Path to save the trained model.", default='./chk/trained_model.pth')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm = NumericLMWrapper(config['model_name'], project_input=False, project_output=True, device=device)
    llm.configure_trainable_layers(train_input_projection=False, train_output_projection=True, train_transformer=False)

    # Ensure the tokenizer has a pad token set
    if llm.tokenizer.pad_token is None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token

    alignment(llm, config, config['num_epochs'], args.model_path, args.savestage2, config['shl'])
