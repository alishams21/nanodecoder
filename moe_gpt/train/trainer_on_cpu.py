"""
CPU-Optimized GPT Trainer for MoE Models

This trainer is specifically designed for CPU training with the following features:

CPU OPTIMIZATIONS:
- Standard PyTorch CPU operations without GPU-specific optimizations
- Simple data loading with standard .to(device) operations
- Compatible with any CPU architecture (x86, ARM, etc.)
- No CUDA dependencies or GPU memory requirements
- Memory management with batch prefetching and memory pooling

REQUIREMENTS:
- CPU: Any modern CPU architecture (x86_64, ARM64, etc.)
- PyTorch: Any version with CPU support
- Memory: Sufficient RAM for model and batch size
- No GPU or CUDA installation required

SUITABILITY:
- Best for: Development, testing, small models, CPU-only environments
- Memory: Uses standard CPU memory management with optimizations
- Performance: Slower than GPU but more accessible
- Compatibility: Works on any system without GPU

NOT SUITABLE FOR:
- Large-scale training (use trainer_on_gpu.py instead)
- Production training with large models
- High-throughput scenarios
- GPU-accelerated environments

For GPU training, use trainer_on_gpu.py with TF32 optimizations.
For TPU training, consider creating a separate trainer_on_tpu.py with XLA optimizations.
"""

import torch
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
import math
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gpt import GPT
from utils.training_utils import load_config, get_lr, get_batch, estimate_loss
from utils.plot_utils import plot_losses
from utils.initialization_utils import apply_gpt2_residual_scaling
from utils.params_util import print_model_info, get_num_params
from utils.memory_utils import memory_optimized_training, MemoryMonitor, optimize_memory_settings
from utils.hellaswag_utils import HellaSwagEvalLoader, evaluate_hellaswag, download_hellaswag_data, setup_hellaswag_evaluation, run_hellaswag_evaluation, run_final_hellaswag_evaluation, should_run_hellaswag_evaluation
import wandb

from utils.vocab_utils import load_and_update_vocab_size
from utils.checkpoint_utils import resume_from_checkpoint, load_optimizer_from_checkpoint
from utils.wandb_utils import setup_wandb_logging, log_training_metrics, finish_wandb_run, log_model_info


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
HELLASWAG_SETTING = config["hellaswag"]  # Add HellaSwag settings
WANDB_SETTING = config.get("wandb")  # Add wandb settings
RUN_TYPE = config["run_type"]
torch.manual_seed(TRAINING_SETTING["seed"])

data_dir = DATA_SETTINGS["dataset"]
device = torch.device(DEVICE_SETTING["device_type"])

# Create a wrapper function for get_batch with the specific parameters
def get_batch_wrapper(split):
    return get_batch(split, data_dir, MODEL_SETTING, TRAINING_SETTING, device, DEVICE_SETTING["device_type"])

# get vocab size
vocab_size = load_and_update_vocab_size(data_dir, MODEL_SETTING)


if RUN_TYPE == 'resume':
    checkpoint, iter_num, best_val_loss = resume_from_checkpoint(
        OUTPUT_SETTINGS['checkpoint_path'], MODEL_SETTING, device
    )
    # create the model
    model = GPT(MODEL_SETTING)
    model.load_state_dict(checkpoint['model'])
    
    # Create optimizer and load its state if available
    optimizer = model.configure_optimizers(OPTIMIZER_SETTING, DEVICE_SETTING["device_type"])
    load_optimizer_from_checkpoint(optimizer, checkpoint)
else:
    # create a new model from scratch
    model = GPT(MODEL_SETTING)
    optimizer = model.configure_optimizers(OPTIMIZER_SETTING, DEVICE_SETTING["device_type"])
    iter_num = 0
    best_val_loss = 1e9
    apply_gpt2_residual_scaling(model, MODEL_SETTING)

# Print detailed model information
print("=" * 50)
print("MODEL INFORMATION")
print("=" * 50)
print_model_info(model, non_embedding=True)
print(f"Model created with {get_num_params(model)/1e6:.2f}M parameters")
print(f"Wandb logging: {WANDB_SETTING.get('enabled', False)}")
print("=" * 50)

# Set device
device = torch.device(DEVICE_SETTING["device_type"])
model.to(device)
print(f"Using device: {device}")

# Initialize memory monitor
memory_monitor = MemoryMonitor(device)

# Initialize HellaSwag evaluation if enabled
hellaswag_eval_loader, HELLASWAG_SETTING = setup_hellaswag_evaluation(HELLASWAG_SETTING, MODEL_SETTING)

# Get memory settings from config
memory_settings = {}

# Auto-optimize settings if enabled
if MEMORY_SETTING["auto_optimize"]:
    auto_settings = optimize_memory_settings(device, TRAINING_SETTING["batch_size"], 
                                           get_num_params(model) * 4 / 1024 / 1024)
    memory_settings.update(auto_settings)
else:
    memory_settings.update(MEMORY_SETTING)


# wandb logging setup
wandb_enabled = setup_wandb_logging(
    WANDB_SETTING, MODEL_SETTING, TRAINING_SETTING, 
    OPTIMIZER_SETTING, LR_SCHEDULE_SETTING, DEVICE_SETTING, 
    HELLASWAG_SETTING, "cpu-training"
)

# Log model information
if wandb_enabled:
    log_model_info(model, WANDB_SETTING)

# init training state
def cpu_based_trainer(model, optimizer, device, max_iters, eval_interval, log_interval, 
                       grad_clip, iter_num, best_val_loss, eval_only=False, checkpoint_interval=None):
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
        hellaswag_accuracies: List of HellaSwag accuracies at each evaluation
    """
    # Initialize tracking lists
    train_losses, val_losses, track_tokens_seen = [], [], []
    hellaswag_accuracies = []  # Track HellaSwag accuracy over time
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
                
            if iter_num % checkpoint_interval == 0:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': MODEL_SETTING,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'memory_stats': memory_stats,
                        'hellaswag_accuracy': hellaswag_accuracies[-1],
                    }
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
                
                # HellaSwag evaluation
                if should_run_hellaswag_evaluation(HELLASWAG_SETTING, iter_num):
                    hellaswag_accuracy = run_hellaswag_evaluation(
                        hellaswag_eval_loader, model, 
                        HELLASWAG_SETTING, 
                        max_batches=2 # Just 2 batches
                    )
                    hellaswag_accuracies.append(hellaswag_accuracy)  # Track accuracy

                memory_monitor.update_peak()
                memory_stats = memory_monitor.get_stats()
                print(f"Iter {iter_num}: train loss {lossf:.4f}, val loss {val_lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, memory {memory_stats['current_mb']:.1f}MB, hellaswag_accuracy {hellaswag_accuracy}")
                
                # wandb logging
                if WANDB_SETTING.get("enabled", False):
                    log_training_metrics(
                        iter_num, losses['train'], losses['val'], 
                        get_lr(iter_num, LR_SCHEDULE_SETTING, OPTIMIZER_SETTING),
                        hellaswag_accuracy, WANDB_SETTING
                    )
            
            iter_num += 1

            # termination
            if iter_num > max_iters:
                break

    print("Training completed!")
    return train_losses, val_losses, track_tokens_seen, hellaswag_accuracies

# Train the model
train_losses, val_losses, tokens_seen, hellaswag_accuracies = cpu_based_trainer(
    model, optimizer, device,
    max_iters=TRAINING_SETTING["max_iters"],
    eval_interval=TRAINING_SETTING["eval_interval"],
    log_interval=TRAINING_SETTING["log_interval"],
    grad_clip=TRAINING_SETTING["grad_clip"],
    iter_num=iter_num,
    best_val_loss=best_val_loss,
    eval_only=TRAINING_SETTING["eval_only"],
    checkpoint_interval=TRAINING_SETTING["checkpoint_interval"]
)

# Plot the training results
if len(train_losses) > 0:
    print("Generating loss plot...")
    iterations_tensor = torch.linspace(0, TRAINING_SETTING["max_iters"], len(train_losses))
    plot_losses(iterations_tensor, tokens_seen, train_losses, val_losses, hellaswag_accuracies)
    plt.savefig(OUTPUT_SETTINGS["loss_plot_path"])
    print(f"Loss plot saved to {OUTPUT_SETTINGS['loss_plot_path']}")

# Save final model
print("Saving final model...")
final_memory_stats = memory_monitor.get_stats()


final_model_data = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "model_args": MODEL_SETTING,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "tokens_seen": tokens_seen,
    "memory_stats": final_memory_stats,
}

torch.save(final_model_data, OUTPUT_SETTINGS["model_save_path"])
print(f"Final model saved to {OUTPUT_SETTINGS['model_save_path']}")

# Finish wandb run
finish_wandb_run(WANDB_SETTING)
