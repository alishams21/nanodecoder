import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.classification_head_utils import (download_and_unzip_spam_data, 
                                                 create_balanced_dataset, 
                                                 random_split, 
                                                 SpamDataset)
from pathlib import Path
import pandas as pd
import torch
import tiktoken
from torch.utils.data import DataLoader
import time
import yaml
from src.utils.gpt2_weights_loader import download_and_load_gpt2, load_weights_into_gpt
from src.models.gpt import GPT
from src.utils.classification_head_utils import calc_loss_batch, calc_accuracy_loader, evaluate_model


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


config = load_config()

# Initialize tokenizer and device
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data configuration
data_config = config["data"]
url = data_config["url"]
zip_path = data_config["zip_path"]
extracted_path = data_config["extracted_path"]
data_file_path = Path(data_config["data_file_path"])

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# Get split ratios from config
train_ratio = data_config["train_ratio"]
validation_ratio = data_config["validation_ratio"]
train_df, validation_df, test_df = random_split(balanced_df, train_ratio, validation_ratio)

# Get output file names from config
output_config = config["output"]
train_df.to_csv(output_config["train_csv"], index=None)
validation_df.to_csv(output_config["validation_csv"], index=None)
test_df.to_csv(output_config["test_csv"], index=None)

train_dataset = SpamDataset(
    csv_file=output_config["train_csv"],
    max_length=data_config["max_length"],
    tokenizer=tokenizer
)
val_dataset = SpamDataset(
    csv_file=output_config["validation_csv"],
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file=output_config["test_csv"],
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# Get training configuration
training_config = config["training"]
num_workers = training_config["num_workers"]
batch_size = training_config["batch_size"]

torch.manual_seed(training_config["seed"])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# Get model selection from config
model_selection = config["model_selection"]
CHOOSE_MODEL = model_selection["choose_model"]
INPUT_PROMPT = model_selection["input_prompt"]

# Get model configuration
BASE_CONFIG = get_model_config(config, CHOOSE_MODEL)

assert train_dataset.max_length <= BASE_CONFIG["max_context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['max_context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['max_context_length']}`"
)


# Get model size for downloading weights
model_size = config["gpt2_models"][CHOOSE_MODEL]["name"].split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPT(BASE_CONFIG)
model = load_weights_into_gpt(model, params)

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(training_config["seed"])

num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["n_embd"], out_features=num_classes)
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True
    
model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(training_config["seed"]) # For reproducibility due to the shuffling in the training data loader

start_time = time.time()

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=training_config["learning_rate"], 
    weight_decay=training_config["weight_decay"]
)

num_epochs = training_config["num_epochs"]




def spam_head_fine_tuning(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen, model, optimizer


train_losses, val_losses, train_accs, val_accs, examples_seen, model, optimizer = spam_head_fine_tuning(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, 
    eval_freq=training_config["eval_freq"], 
    eval_iter=training_config["eval_iter"],
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Save the trained model and optimizer
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, 
output_config["model_save_path"]
)






