import torch
import tiktoken
import matplotlib.pyplot as plt
import yaml
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gpt import GPT
from utils.dataloader import create_sliding_window_dataloader
from utils.training_utils import calc_loss_batch, evaluate_model
from utils.plot_utils import plot_losses


def load_config(config_path="src/train/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config()
MODEL_SETTING = config["model"]
TRAINING_SETTING = config["training"]
DATA_SETTINGS = config["data"]
OUTPUT_SETTINGS = config["output"]
torch.manual_seed(TRAINING_SETTING["seed"])

#load data
file_path = DATA_SETTINGS["file_path"]
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

train_ratio = DATA_SETTINGS["train_ratio"]
split_idx = int(train_ratio * len(text_data))

train_loader = create_sliding_window_dataloader(
    text_data[:split_idx],
    batch_size=TRAINING_SETTING["batch_size"],
    max_length=MODEL_SETTING["max_context_length"],
    stride=MODEL_SETTING["max_context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_sliding_window_dataloader(
    text_data[split_idx:],
    batch_size=TRAINING_SETTING["batch_size"],
    max_length=MODEL_SETTING["max_context_length"],
    stride=MODEL_SETTING["max_context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


#load model
model = GPT(MODEL_SETTING)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#load optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=TRAINING_SETTING["learning_rate"], weight_decay=TRAINING_SETTING["weight_decay"]
)


def train_custom_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

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


    return train_losses, val_losses, track_tokens_seen


train_losses, val_losses, tokens_seen = train_custom_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=TRAINING_SETTING["num_epochs"], 
    eval_freq=DATA_SETTINGS["eval_freq"], 
    eval_iter=DATA_SETTINGS["eval_iter"]
)

epochs_tensor = torch.linspace(0, TRAINING_SETTING["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
plt.savefig(OUTPUT_SETTINGS["loss_plot_path"])

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    OUTPUT_SETTINGS["model_save_path"]
)
    


