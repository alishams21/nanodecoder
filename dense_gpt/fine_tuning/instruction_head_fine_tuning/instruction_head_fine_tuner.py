import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import json
import os
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
from src.utils.instruction_head_utils import custom_collate_fn, InstructionDataset
from src.utils.plot_utils import plot_losses
import matplotlib.pyplot as plt
import torch
import tiktoken
from src.utils.gpt2_weights_loader import download_and_load_gpt2
from src.models.gpt import GPT
from src.utils.gpt2_weights_loader import load_weights_into_gpt
from src.utils.training_utils import calc_loss_batch, evaluate_model

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_model_config(config, model_key):
    """Get specific model configuration from config."""
    model_config = config["gpt2_models"][model_key].copy()
    
    # Convert to the format expected by GPT model
    gpt_config = {
        "vocab_size": model_config["vocab_size"],
        "max_context_length": model_config["context_length"],
        "n_embd": model_config["n_embd"],
        "n_heads": model_config["n_heads"],
        "n_blocks": model_config["n_blocks"],
        "drop_rate": model_config["drop_rate"],
        "qkv_bias": model_config["qkv_bias"]
    }
    
    return gpt_config

def load_instruction_data(file_path):
    """Load instruction data from local JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instruction data file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data


# Load configuration
config = load_config()

# Get data configuration
data_config = config["data"]
file_path = data_config["file_path"]
data = load_instruction_data(file_path)
print("Number of entries:", len(data))

# Get split ratios from config
train_ratio = data_config["train_ratio"]
train_portion = int(len(data) * train_ratio)
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model selection from config
model_selection = config["model_selection"]
CHOOSE_MODEL = model_selection["choose_model"]

# Get model configuration
BASE_CONFIG = get_model_config(config, CHOOSE_MODEL)

# Get training configuration
training_config = config["training"]
num_workers = training_config["num_workers"]
batch_size = training_config["batch_size"]

# Set up collate function with config max length
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=BASE_CONFIG["max_context_length"]
)

torch.manual_seed(training_config["seed"])

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# Get model size for downloading weights
model_size = config["gpt2_models"][CHOOSE_MODEL]["name"].split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPT(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=training_config["learning_rate"], 
    weight_decay=training_config["weight_decay"]
)

num_epochs = training_config["num_epochs"]

def instruction_head_fine_tuning(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )

    return train_losses, val_losses, track_tokens_seen

train_losses, val_losses, tokens_seen = instruction_head_fine_tuning(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, 
    eval_freq=training_config["eval_freq"], 
    eval_iter=training_config["eval_iter"],
)

# Get output configuration
output_config = config["output"]

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

plt.savefig(output_config["loss_plot_path"])

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, 
output_config["model_save_path"]
)