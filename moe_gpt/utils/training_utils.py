import torch
import yaml
import os
import pickle
import math
import numpy as np
import torch

def load_config(config_path="moe_gpt/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_lr(it, lr_schedule_setting, optimizer_setting):
    """Learning rate scheduler with cosine decay and warmup."""
    if it < lr_schedule_setting["warmup_iters"]:
        return optimizer_setting["learning_rate"] * (it + 1) / (lr_schedule_setting["warmup_iters"] + 1)
    if it > lr_schedule_setting["lr_decay_iters"]:
        return lr_schedule_setting["min_lr"]
    decay_ratio = (it - lr_schedule_setting["warmup_iters"]) / (lr_schedule_setting["lr_decay_iters"] - lr_schedule_setting["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_schedule_setting["min_lr"] + coeff * (optimizer_setting["learning_rate"] - lr_schedule_setting["min_lr"])


def get_batch(split, data_dir, model_setting, training_setting, device, device_type='cpu'):
    """Get a batch of data for training/validation."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - model_setting["max_context_length"], (training_setting["batch_size"],))
    x = torch.stack([torch.from_numpy((data[i:i+model_setting["max_context_length"]]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+model_setting["max_context_length"]]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, get_batch_func, training_setting):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(training_setting["eval_iters"])
        for k in range(training_setting["eval_iters"]):
            X, Y = get_batch_func(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


