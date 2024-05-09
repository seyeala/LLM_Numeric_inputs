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
    targets = torch.rand(batch_size, 1).to(device)
    return tensor_inputs, targets

def alignment(llm, config, num_epochs, load_model_path, save_model_path, shl):
    device = next(llm.parameters()).device
    optimizer = Adam(filter(lambda p: p.requires_grad, llm.parameters()), lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if shl else None
    scaler = GradScaler()

    # Load model state if exists
    if load_model_path:
        llm.load_state_dict(torch.load(load_model_path))
        print(f"Loaded model from {load_model_path}")

    llm.train()

    for epoch in range(num_epochs):
        epoch_loss = 0

        for _ in range(config['num_batches']):
            batch_inputs, batch_targets = generate_text_data(config['batch_size'], config['min_val'], config['max_val'], device, llm.tokenizer)

            optimizer.zero_grad()
            with autocast():
                outputs = llm(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)
                epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if shl:
            scheduler.step()

        print(f'Epoch {epoch+1}: Average Loss {epoch_loss / config['num_batches']}')

    torch.save(llm.state_dict(), save_model_path)
    print(f"Saved trained model to {save_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with text inputs.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--save_model_path", help="Path to save the trained model.")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm = NumericLMWrapper(config['model_name'], project_input=False, project_output=True, train_transformer=False, device=device)
    llm.configure_trainable_layers(train_input_projection=False, train_output_projection=True, train_transformer=False)

    alignment(llm, config, config['num_epochs'], args.model_path, './chk/stage.path', config['shl'])
