"""
GPU-Optimized GPT Trainer for MoE Models

This trainer is specifically optimized for GPU training with the following features:

GPU OPTIMIZATIONS:
- TF32 (Tensor Float 32-bit) support for ~1.5x faster matrix operations
- cuDNN TF32 optimization for faster convolutions and neural network operations
- CUDA memory pinning for faster CPU-GPU data transfer
- Non-blocking GPU transfers for improved pipeline efficiency
- Memory management with batch prefetching and memory pooling

REQUIREMENTS:
- GPU: Ampere architecture (RTX 30xx, A100, etc.) for TF32 support
- PyTorch: 1.7+ for TF32 compatibility
- CUDA: 11.0+ for TF32 operations
- Memory: Sufficient GPU memory for model and batch size

SUITABILITY:
- Best for: Large MoE models, production training, high-throughput scenarios
- Memory: Optimized for GPU memory management with pin_memory() and memory pooling
- Performance: ~1.5x speedup over standard FP32 training
- Accuracy: ~99.9% accuracy retention with TF32 optimizations

NOT SUITABLE FOR:
- CPU-only environments (use trainer_on_cpu.py instead)
- TPU training (requires different optimizations)
- Older GPUs without TF32 support
- Memory-constrained environments

For TPU training, consider creating a separate trainer_on_tpu.py with XLA optimizations.
"""

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
from utils.plot_utils import plot_losses
from utils.initialization_utils import apply_gpt2_residual_scaling
from utils.params_util import print_model_info, get_num_params
from utils.training_utils import load_config, get_lr, get_batch, estimate_loss
from utils.memory_utils import memory_optimized_training, MemoryMonitor, optimize_memory_settings

import numpy as np


# Load configuration
config = load_config()
MODEL_SETTING = config["model"]
TRAINING_SETTING = config["training"]
DATA_SETTINGS = config["data"]
OUTPUT_SETTINGS = config["output"]
OPTIMIZER_SETTING = config["optimizer"]
LR_SCHEDULE_SETTING = config["lr_schedule"]
DEVICE_SETTING = config["device"]
MEMORY_SETTING = config["memory"]  # Add memory settings
RUN_TYPE = config["run_type"]
torch.manual_seed(TRAINING_SETTING["seed"])

data_dir = DATA_SETTINGS["dataset"]
device = torch.device(DEVICE_SETTING["device_type"])
# enable optimizations for GPU
if DEVICE_SETTING["device_type"] == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda'
else:
    device_type = 'cpu'


# Create a wrapper function for get_batch with the specific parameters
def get_batch_wrapper(split):
    return get_batch(split, data_dir, MODEL_SETTING, TRAINING_SETTING, device, device_type)

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

if RUN_TYPE=='resume':
    print(f"Resuming training from {OUTPUT_SETTINGS['checkpoint_path']}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(OUTPUT_SETTINGS['checkpoint_path'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_blocks', 'n_head', 'n_embd', 'max_context_length', 'bias', 'vocab_size']:
        MODEL_SETTING[k] = checkpoint_model_args[k]
    # create the model
    model = GPT(MODEL_SETTING)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    # unwanted_prefix = '_orig_mod.'
    # for k,v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    # Create optimizer and load its state if available
    optimizer = model.configure_optimizers(OPTIMIZER_SETTING, DEVICE_SETTING["device_type"])
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
else:
    # create a new model from scratch
    model = GPT(MODEL_SETTING)
    optimizer = model.configure_optimizers(OPTIMIZER_SETTING, DEVICE_SETTING["device_type"])
    iter_num = 0
    best_val_loss = 1e9
    apply_gpt2_residual_scaling(model, MODEL_SETTING)
    # Apply GPT-2 residual scaling for better initialization (only for new models)
# Print detailed model information
print("=" * 50)
print("MODEL INFORMATION")
print("=" * 50)
print_model_info(model, non_embedding=True)
print(f"Model created with {get_num_params(model)/1e6:.2f}M parameters")
print("=" * 50)

# Set device
device = torch.device(DEVICE_SETTING["device_type"])    
model.to(device)
print(f"Using device: {device}")

# Initialize memory monitor
memory_monitor = MemoryMonitor(device)

# Get memory settings from config
memory_settings = {
    'buffer_size': MEMORY_SETTING["buffer_size"],
    'enable_memory_pool': MEMORY_SETTING["enable_memory_pool"],
    'max_pool_size': MEMORY_SETTING["max_pool_size"],
    'enable_prefetching': MEMORY_SETTING["enable_prefetching"],
    'prefetch_timeout': MEMORY_SETTING["prefetch_timeout"],
    'memory_monitoring': MEMORY_SETTING["memory_monitoring"]
}

# Auto-optimize settings if enabled
if MEMORY_SETTING["auto_optimize"]:
    auto_settings = optimize_memory_settings(device, TRAINING_SETTING["batch_size"], 
                                           get_num_params(model) * 4 / 1024 / 1024)
    memory_settings.update(auto_settings)

print(f"Memory optimization settings: {memory_settings}")



def gpu_based_trainer(model, optimizer, device, max_iters, eval_interval, log_interval, 
                       grad_clip, iter_num, best_val_loss, eval_only=False):
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
    
    # Create checkpoint directory
    os.makedirs(OUTPUT_SETTINGS["checkpoint_path"], exist_ok=True)
    
    print("Starting training...")
    
    # Use memory-optimized training
    with memory_optimized_training(
        get_batch_wrapper, 
        device, 
        buffer_size=memory_settings['buffer_size'],
        enable_memory_pool=memory_settings['enable_memory_pool']
    ) as async_loader:
        
        X, Y = async_loader.get_batch()
        t0 = time.time()

        while True:
            # set learning rate
            lr = get_lr(iter_num, LR_SCHEDULE_SETTING, OPTIMIZER_SETTING) if LR_SCHEDULE_SETTING["decay_lr"] else OPTIMIZER_SETTING["learning_rate"]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate
            if iter_num % eval_interval == 0:
                losses = estimate_loss(model, get_batch_wrapper, TRAINING_SETTING)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                track_tokens_seen.append(tokens_seen)
                
                # Update memory monitoring
                memory_monitor.update_peak()
                memory_stats = memory_monitor.get_stats()
                
                print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if device.type == 'cuda':
                    print(f"GPU memory: {memory_stats['current_mb']:.1f}MB (peak: {memory_stats['peak_mb']:.1f}MB)")
                else:
                    print(f"Memory usage: {memory_stats['current_mb']:.1f}MB (peak: {memory_stats['peak_mb']:.1f}MB)")
                
                if losses['val'] < best_val_loss or OUTPUT_SETTINGS["always_save_checkpoint"]:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': MODEL_SETTING,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'memory_stats': memory_stats,
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_SETTING["grad_clip"])
            
            # optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # get next batch using memory-optimized loader
            X, Y = async_loader.get_batch()
            tokens_seen += X.numel()

            # logging
            if iter_num % log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                lossf = loss.item()
                
                # Compute validation loss for logging
                model.eval()
                with torch.no_grad():
                    val_X, val_Y = get_batch_wrapper('val')
                    _, val_loss = model(val_X, val_Y)
                    val_lossf = val_loss.item()
                model.train()
                
                memory_stats = memory_monitor.get_stats()
                if device.type == 'cuda':
                    print(f"Iter {iter_num}: train loss {lossf:.4f}, val loss {val_lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, GPU memory {memory_stats['current_mb']:.1f}MB")
                else:
                    print(f"Iter {iter_num}: train loss {lossf:.4f}, val loss {val_lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, memory {memory_stats['current_mb']:.1f}MB")
            
            iter_num += 1

            # termination
            if iter_num > max_iters:
                break

    print("Training completed!")
    return train_losses, val_losses, track_tokens_seen

# Train the model
train_losses, val_losses, tokens_seen = gpu_based_trainer(
    model, optimizer, device,
    max_iters=TRAINING_SETTING["max_iters"],
    eval_interval=TRAINING_SETTING["eval_interval"],
    log_interval=TRAINING_SETTING["log_interval"],
    grad_clip=TRAINING_SETTING["grad_clip"],
    iter_num=iter_num,
    best_val_loss=best_val_loss,
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
final_memory_stats = memory_monitor.get_stats()
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "model_args": MODEL_SETTING,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "tokens_seen": tokens_seen,
    "memory_stats": final_memory_stats,
}, OUTPUT_SETTINGS["model_save_path"])
print(f"Final model saved to {OUTPUT_SETTINGS['model_save_path']}")
print(f"Final memory stats: {final_memory_stats}")
