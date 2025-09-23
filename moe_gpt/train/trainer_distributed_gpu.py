"""
Distributed Mixed Precision GPU-Optimized GPT Trainer for MoE Models with Model Compilation

This trainer combines the best of all worlds:
- Advanced GPU optimizations from trainer_on_gpu.py
- Mixed precision training from from_scratch_mixed_precision.py
- Distributed Data Parallel (DDP) from from_scratch_ddp.py
- Model compilation from from_scratch_compile.py
- Wandb logging for experiment tracking

FEATURES:
- Distributed Data Parallel (DDP) for multi-GPU training
- Mixed Precision Training: bfloat16/float16 with automatic detection
- TF32 (Tensor Float 32-bit) support for ~1.5x faster matrix operations
- cuDNN TF32 optimization for faster convolutions and neural network operations
- CUDA memory pinning for faster CPU-GPU data transfer
- Non-blocking GPU transfers for improved pipeline efficiency
- Memory management with batch prefetching and memory pooling
- YAML configuration management
- Advanced model initialization with GPT-2 residual scaling
- Visualization and plotting capabilities
- Comprehensive memory monitoring
- Gradient accumulation for large effective batch sizes
- Multi-GPU synchronization and coordination
- PyTorch 2.0 Model Compilation for additional speedup
- Wandb experiment tracking and logging

REQUIREMENTS:
- Multiple GPUs: 2+ GPUs for DDP training
- GPU: Ampere architecture (RTX 30xx, A100, etc.) for TF32 support
- PyTorch: 2.0+ for model compilation support
- CUDA: 11.0+ for TF32 operations
- NCCL: For multi-GPU communication
- Memory: Sufficient GPU memory for model and batch size per GPU
- Wandb: For experiment tracking (optional)

SUITABILITY:
- Best for: Large MoE models, production training, high-throughput scenarios
- Memory: Optimized for GPU memory management with mixed precision
- Performance: ~1.5x speedup over standard FP32 training + memory savings from mixed precision + linear scaling with GPUs + additional compilation speedup
- Accuracy: ~99.9% accuracy retention with TF32 + mixed precision optimizations
- Scalability: Linear scaling with number of GPUs (2x GPUs = ~2x speed)
- Compilation: Additional 1.2-1.5x speedup from PyTorch 2.0 compilation

NOT SUITABLE FOR:
- CPU-only environments (use trainer_on_cpu.py instead)
- Single GPU training (use trainer_mixed_precision.py instead)
- TPU training (requires different optimizations)
- Older GPUs without TF32 support
- Memory-constrained environments without multiple GPUs
- PyTorch < 2.0 (compilation requires PyTorch 2.0+)
"""

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import tiktoken
import matplotlib.pyplot as plt
import yaml
import sys
import os
import pickle
import time
import math
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gpt import GPT
from utils.plot_utils import plot_losses
from utils.initialization_utils import apply_gpt2_residual_scaling
from utils.params_util import print_model_info, get_num_params
from utils.training_utils import load_config, get_lr, get_batch, estimate_loss
from utils.memory_utils import memory_optimized_training, MemoryMonitor, optimize_memory_settings

import numpy as np
import wandb

# Load configuration
config = load_config()
MODEL_SETTING = config["model"]
TRAINING_SETTING = config["training"]
DATA_SETTINGS = config["data"]
OUTPUT_SETTINGS = config["output"]
OPTIMIZER_SETTING = config["optimizer"]
LR_SCHEDULE_SETTING = config["lr_schedule"]
DEVICE_SETTING = config["device"]
MEMORY_SETTING = config["memory"]
COMPILE_SETTING = config.get("compile", {"enabled": True})  # Default to enabled if not specified
WANDB_SETTING = config.get("wandb")
torch.manual_seed(TRAINING_SETTING["seed"])

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
nproc_per_node = DEVICE_SETTING.get("nproc_per_node", 1)

# Auto-launch distributed training if nproc_per_node > 1 and not already in DDP mode
if not ddp and nproc_per_node > 1:
    available_gpus = torch.cuda.device_count()
    if available_gpus >= nproc_per_node:
        print(f"Auto-launching distributed training with {nproc_per_node} GPUs...")
        import subprocess
        import sys
        
        # Launch torchrun with the configured number of processes
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={nproc_per_node}",
            "--use_env",
            __file__
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        sys.exit(0)
    else:
        print(f"Warning: Requested {nproc_per_node} GPUs but only {available_gpus} available. Falling back to single GPU training.")

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = nproc_per_node
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mixed precision configuration
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# -----------------------------------------------------------------------------
if master_process:
    print("=" * 50)
    print("DISTRIBUTED MIXED PRECISION GPU TRAINING SCRIPT WITH COMPILATION")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"Data type: {dtype}")
    print(f"Model compilation: {COMPILE_SETTING.get('enabled', True)}")
    print(f"Batch size: {TRAINING_SETTING['batch_size']}")
    print(f"Model size: {MODEL_SETTING['n_blocks']} layers, {MODEL_SETTING['n_head']} heads, {MODEL_SETTING['n_embd']} embd")
    print(f"DDP world size: {ddp_world_size}")
    print(f"Configured nproc_per_node: {nproc_per_node}")
    if nproc_per_node > 1:
        print(f"Auto-launch: Will use {nproc_per_node} GPUs when available")
    print(f"Wandb logging: {WANDB_SETTING.get('enabled', False)}")
    print("=" * 50)

# Enable optimizations for GPU
if DEVICE_SETTING["device_type"] == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda'
else:
    device_type = 'cpu'

# Mixed precision context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('gpt_moe', DATA_SETTINGS["dataset"])

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
    if master_process:
        print(f"Found vocab_size = {meta_vocab_size}")
    
vocab_size = meta_vocab_size if meta_vocab_size is not None else MODEL_SETTING["vocab_size"]
# Update vocab_size in MODEL_SETTING if we got it from meta
if meta_vocab_size is not None:
    MODEL_SETTING["vocab_size"] = meta_vocab_size

model = GPT(MODEL_SETTING)

# Apply GPT-2 residual scaling for better initialization
apply_gpt2_residual_scaling(model, MODEL_SETTING)

# Print detailed model information
if master_process:
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print_model_info(model, non_embedding=True)
    print(f"Model created with {get_num_params(model)/1e6:.2f}M parameters")
    print("=" * 50)

optimizer = model.configure_optimizers(OPTIMIZER_SETTING, DEVICE_SETTING["device_type"])

# Set device
device = torch.device(device)    
model.to(device)
if master_process:
    print(f"Using device: {device}")

# Initialize GradScaler for mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

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

if master_process:
    print(f"Memory optimization settings: {memory_settings}")

# Wrap model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Compile the model for faster training (PyTorch 2.0+)
if COMPILE_SETTING.get("enabled", True) and master_process:
    print("Compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0
    print("Model compilation completed!")

# init training state
iter_num = 0

# wandb logging setup
if WANDB_SETTING.get("enabled", False) and master_process:
    try:
        # Generate run name with timestamp if not specified
        run_name = WANDB_SETTING.get("run_name")
        run_name = f"{run_name}-{int(time.time())}"
        
        # Create wandb config
        wandb_config = {
            'n_layer': MODEL_SETTING['n_blocks'],
            'n_head': MODEL_SETTING['n_head'], 
            'n_embd': MODEL_SETTING['n_embd'],
            'n_exp': MODEL_SETTING.get('n_exp', 0),
            'top_k': MODEL_SETTING.get('top_k', 1),
            'use_aux_loss': MODEL_SETTING.get('use_aux_loss', False),
            'use_router_z_loss': MODEL_SETTING.get('use_router_z_loss', False),
            'learning_rate': OPTIMIZER_SETTING['learning_rate'],
            'batch_size': TRAINING_SETTING['batch_size'],
            'block_size': MODEL_SETTING['max_context_length'],
            'dtype': dtype,
            'compile': COMPILE_SETTING.get('enabled', True),
            'ddp_world_size': ddp_world_size,
            'accumulation_steps': TRAINING_SETTING.get('accumulation_steps', 1),
            'grad_clip': TRAINING_SETTING.get('grad_clip', 1.0),
            'weight_decay': OPTIMIZER_SETTING.get('weight_decay', 0.1),
            'warmup_iters': LR_SCHEDULE_SETTING.get('warmup_iters', 400),
            'lr_decay_iters': LR_SCHEDULE_SETTING.get('lr_decay_iters', 700),
            'min_lr': LR_SCHEDULE_SETTING.get('min_lr', 0.0001),
        }
        
        wandb.init(
            project=WANDB_SETTING.get("project"),
            name=run_name,
            config=wandb_config
        )
        print(f"Wandb initialized: {run_name}")
    except ImportError:
        print("Warning: wandb not installed. Install with: pip install wandb")
        WANDB_SETTING["enabled"] = False
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        WANDB_SETTING["enabled"] = False

def distributed_mixed_precision_trainer(model, optimizer, device, max_iters, eval_interval, log_interval, 
                                       grad_clip, accumulation_steps, eval_only=False):
    """
    Train the model with distributed mixed precision and GPU optimizations.
    
    Args:
        model: The GPT model to train (wrapped in DDP if distributed)
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
    best_val_loss = TRAINING_SETTING["best_val_loss"]
    
    # Gradient accumulation variables
    accumulation_count = 0
    accumulated_loss = 0.0
    
    # DDP gradient sync optimization
    if ddp:
        # Scale accumulation steps per process for DDP
        accumulation_steps_per_process = accumulation_steps // ddp_world_size
        if accumulation_steps_per_process == 0:
            accumulation_steps_per_process = 1
    else:
        accumulation_steps_per_process = accumulation_steps
    
    # Create checkpoint directory
    if master_process:
        os.makedirs(OUTPUT_SETTINGS["checkpoint_path"], exist_ok=True)
    
    if master_process:
        print("Starting distributed mixed precision training...")
        if accumulation_steps_per_process > 1:
            print(f"Gradient accumulation enabled: {accumulation_steps_per_process} steps")
            print(f"Per-process accumulation steps: {accumulation_steps_per_process}")
            print(f"Effective batch size: {TRAINING_SETTING['batch_size'] * accumulation_steps * ddp_world_size}")
            if ddp:
                print(f"DDP gradient sync optimization: Only sync at last micro step")
        else:
            print("No gradient accumulation (accumulation_steps = 1)")
    
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
            if iter_num % eval_interval == 0 and master_process:
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
                
                # wandb logging
                if WANDB_SETTING.get("enabled", False):
                    try:
                        
                        wandb.log({
                            "iter": iter_num,
                            "train/loss": losses['train'],
                            "val/loss": losses['val'],
                            "lr": lr,
                            "gpu_memory_mb": memory_stats['current_mb'],
                            "peak_memory_mb": memory_stats['peak_mb'],
                            "tokens_seen": tokens_seen,
                        })
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")
                
                if losses['val'] < best_val_loss or OUTPUT_SETTINGS["always_save_checkpoint"]:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        # Get raw model for saving (unwrap DDP)
                        raw_model = model.module if ddp else model
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': MODEL_SETTING,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'memory_stats': memory_stats,
                            'dtype': dtype,  # Save mixed precision dtype
                            'ddp_world_size': ddp_world_size,  # Save DDP info
                            'compiled': COMPILE_SETTING.get("enabled", True),  # Save compilation info
                        }
                        print(f"Saving checkpoint to {OUTPUT_SETTINGS['checkpoint_path']}")
                        torch.save(checkpoint, os.path.join(OUTPUT_SETTINGS["checkpoint_path"], 'ckpt.pt'))
            
            if iter_num == 0 and eval_only:
                break

            # forward pass with mixed precision
            with ctx:
                logits, loss = model(X, Y)
            
            # Scale loss by accumulation steps for gradient accumulation
            loss = loss / accumulation_steps_per_process
            accumulated_loss += loss.item()
            
            # backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation logic
            accumulation_count += 1
            
            # DDP gradient sync optimization
            if ddp:
                # Only sync gradients at the last micro step
                model.require_backward_grad_sync = (accumulation_count % accumulation_steps_per_process == 0)
            
            # Only update model after accumulating enough gradients
            if accumulation_count % accumulation_steps_per_process == 0:
                # gradient clipping
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                # optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Reset accumulation
                accumulation_count = 0
                accumulated_loss = 0.0

            # get next batch using memory-optimized loader
            X, Y = async_loader.get_batch()
            tokens_seen += X.numel()

            # logging
            if iter_num % log_interval == 0 and master_process:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                # Use accumulated loss if we're in accumulation mode
                lossf = accumulated_loss if accumulation_steps > 1 else loss.item()
                
                # Compute validation loss for logging (using eval_iters for accuracy)
                model.eval()
                with torch.no_grad():
                    val_losses_tensor = torch.zeros(TRAINING_SETTING["eval_iters"])
                    for k in range(TRAINING_SETTING["eval_iters"]):
                        val_X, val_Y = get_batch_wrapper('val')
                        with ctx:  # Use mixed precision context for validation
                            _, val_loss = model(val_X, val_Y)
                        val_losses_tensor[k] = val_loss.item()
                    val_lossf = val_losses_tensor.mean()
                model.train()
                
                memory_stats = memory_monitor.get_stats()
                compile_status = "compiled" if COMPILE_SETTING.get("enabled", True) else "uncompiled"
                if device.type == 'cuda':
                    print(f"Iter {iter_num}: train loss {lossf:.4f}, val loss {val_lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, GPU memory {memory_stats['current_mb']:.1f}MB, dtype {dtype}, world_size {ddp_world_size}, {compile_status}")
                else:
                    print(f"Iter {iter_num}: train loss {lossf:.4f}, val loss {val_lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, memory {memory_stats['current_mb']:.1f}MB, dtype {dtype}, world_size {ddp_world_size}, {compile_status}")
            
            iter_num += 1

            # termination
            if iter_num > max_iters:
                break

    if master_process:
        print("Distributed mixed precision training completed!")
    return train_losses, val_losses, track_tokens_seen

# Train the model
train_losses, val_losses, tokens_seen = distributed_mixed_precision_trainer(
    model, optimizer, device,
    max_iters=TRAINING_SETTING["max_iters"],
    eval_interval=TRAINING_SETTING["eval_interval"],
    log_interval=TRAINING_SETTING["log_interval"],
    grad_clip=TRAINING_SETTING["grad_clip"],
    eval_only=TRAINING_SETTING["eval_only"],
    accumulation_steps=TRAINING_SETTING.get("accumulation_steps", 1)
)

# Plot the training results
if len(train_losses) > 0 and master_process:
    print("Generating loss plot...")
    iterations_tensor = torch.linspace(0, TRAINING_SETTING["max_iters"], len(train_losses))
    plot_losses(iterations_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig(OUTPUT_SETTINGS["loss_plot_path"])
    print(f"Loss plot saved to {OUTPUT_SETTINGS['loss_plot_path']}")

# Save final model
if master_process:
    print("Saving final model...")
    final_memory_stats = memory_monitor.get_stats()
    # Get raw model for saving (unwrap DDP)
    raw_model = model.module if ddp else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_args": MODEL_SETTING,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
        "memory_stats": final_memory_stats,
        "dtype": dtype,  # Save mixed precision dtype
        "scaler_state": scaler.state_dict(),  # Save scaler state for resuming
        "ddp_world_size": ddp_world_size,  # Save DDP info
        "compiled": COMPILE_SETTING.get("enabled", True),  # Save compilation info
    }, OUTPUT_SETTINGS["model_save_path"])
    print(f"Final model saved to {OUTPUT_SETTINGS['model_save_path']}")
    print(f"Final memory stats: {final_memory_stats}")
    compile_status = "compiled" if COMPILE_SETTING.get("enabled", True) else "uncompiled"
    print(f"Training completed with mixed precision dtype: {dtype}, {ddp_world_size} GPUs, and {compile_status} model")

# Clean up DDP
if ddp:
    destroy_process_group()

# Finish wandb run
if WANDB_SETTING.get("enabled", False) and master_process:
    try:
        wandb.finish()
        print("Wandb run finished successfully")
    except Exception as e:
        print(f"Warning: Failed to finish wandb run: {e}")
