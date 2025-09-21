import torch
import tiktoken
import matplotlib.pyplot as plt
import yaml
import sys
import os
import pickle
import time
import math
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gpt import GPT
from utils.dataloader import create_sliding_window_dataloader
from utils.training_utils import calc_loss_batch, evaluate_model
from utils.plot_utils import plot_losses
from utils.initialization_utils import apply_gpt2_residual_scaling
from utils.params_util import print_model_info, get_num_params

import numpy as np


def load_config(config_path="gpt_moe/config.yaml"):
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
OPTIMIZER_SETTING = config["optimizer"]
LR_SCHEDULE_SETTING = config["lr_schedule"]
torch.manual_seed(TRAINING_SETTING["seed"])

data_dir = os.path.join('gpt_moe', DATA_SETTINGS["dataset"])
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - MODEL_SETTING["max_context_length"], (TRAINING_SETTING["batch_size"],))
    x = torch.stack([torch.from_numpy((data[i:i+MODEL_SETTING["max_context_length"]]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+MODEL_SETTING["max_context_length"]]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# get vocab size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size}")
vocab_size = meta_vocab_size if meta_vocab_size is not None else MODEL_SETTING["vocab_size"]
# Update vocab_size in MODEL_SETTING if we got it from meta
if meta_vocab_size is not None:
    MODEL_SETTING["vocab_size"] = meta_vocab_size
model = GPT(MODEL_SETTING)

# Apply GPT-2 residual scaling for better initialization
apply_gpt2_residual_scaling(model, MODEL_SETTING)

# Print detailed model information
print("=" * 50)
print("MODEL INFORMATION")
print("=" * 50)
print_model_info(model, non_embedding=True)
print(f"Model created with {get_num_params(model)/1e6:.2f}M parameters")
print("=" * 50)

optimizer = model.configure_optimizers(OPTIMIZER_SETTING)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(TRAINING_SETTING["eval_iters"])
        for k in range(TRAINING_SETTING["eval_iters"]):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate scheduler
def get_lr(it):
    if it < LR_SCHEDULE_SETTING["warmup_iters"]:
        return OPTIMIZER_SETTING["learning_rate"] * (it + 1) / (LR_SCHEDULE_SETTING["warmup_iters"] + 1)
    if it > LR_SCHEDULE_SETTING["lr_decay_iters"]:
        return LR_SCHEDULE_SETTING["min_lr"]
    decay_ratio = (it - LR_SCHEDULE_SETTING["warmup_iters"]) / (LR_SCHEDULE_SETTING["lr_decay_iters"] - LR_SCHEDULE_SETTING["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LR_SCHEDULE_SETTING["min_lr"] + coeff * (OPTIMIZER_SETTING["learning_rate"] - LR_SCHEDULE_SETTING["min_lr"])

# init training state
iter_num = 0
best_val_loss = 1e9

def cpu_based_trainer(model, optimizer, device, max_iters, eval_interval, log_interval, 
                       grad_clip, eval_only=False):
    """
    Train the model with the given configuration.
    
    Args:
        model: The GPT model to train
        optimizer: The optimizer
        device: Device to train on
        max_iters: Maximum number of training iterations
        eval_interval: Evaluate every N iterations
        log_interval: Log every N iterations
        grad_clip: Gradient clipping value
        eval_only: If True, only evaluate the model
    
    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
        track_tokens_seen: List of tokens seen at each evaluation
    """
    # Initialize tracking lists
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    iter_num = 0
    best_val_loss = 1e9
    
    # Create checkpoint directory
    os.makedirs(OUTPUT_SETTINGS["checkpoint_path"], exist_ok=True)
    
    print("Starting training...")
    X, Y = get_batch('train')
    t0 = time.time()

    while True:
        # set learning rate
        lr = get_lr(iter_num) if LR_SCHEDULE_SETTING["decay_lr"] else OPTIMIZER_SETTING["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            track_tokens_seen.append(tokens_seen)
            
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss or OUTPUT_SETTINGS["always_save_checkpoint"]:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': MODEL_SETTING,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"Saving checkpoint to {OUTPUT_SETTINGS['checkpoint_path']}")
                    torch.save(checkpoint, os.path.join(OUTPUT_SETTINGS["checkpoint_path"], 'ckpt.pt'))
        
        if iter_num == 0 and eval_only:
            break

        # forward pass
        logits, loss = model(X, Y)
        
        # backward pass
        loss.backward()
        
        # gradient clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # get next batch
        X, Y = get_batch('train')
        tokens_seen += X.numel()

        # logging
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item()
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
        
        iter_num += 1

        # termination
        if iter_num > max_iters:
            break

    print("Training completed!")
    return train_losses, val_losses, track_tokens_seen

# Train the model
train_losses, val_losses, tokens_seen = cpu_based_trainer(
    model, optimizer, device,
    max_iters=TRAINING_SETTING["max_iters"],
    eval_interval=TRAINING_SETTING["eval_interval"],
    log_interval=TRAINING_SETTING["log_interval"],
    grad_clip=TRAINING_SETTING["grad_clip"],
    eval_only=TRAINING_SETTING["eval_only"]
)

# Plot the training results
if len(train_losses) > 0:
    print("Generating loss plot...")
    iterations_tensor = torch.linspace(0, TRAINING_SETTING["max_iters"], len(train_losses))
    plot_losses(iterations_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig(OUTPUT_SETTINGS["loss_plot_path"])
    print(f"Loss plot saved to {OUTPUT_SETTINGS['loss_plot_path']}")

# Save final model
print("Saving final model...")
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "model_args": MODEL_SETTING,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "tokens_seen": tokens_seen,
}, OUTPUT_SETTINGS["model_save_path"])
print(f"Final model saved to {OUTPUT_SETTINGS['model_save_path']}")
