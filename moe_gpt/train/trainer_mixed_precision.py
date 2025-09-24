"""
Mixed Precision GPU-Optimized GPT Trainer for MoE Models

This trainer combines the best of both worlds:
- Advanced GPU optimizations from trainer_on_gpu.py
- Mixed precision training from from_scratch_mixed_precision.py

FEATURES:
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

REQUIREMENTS:
- GPU: Ampere architecture (RTX 30xx, A100, etc.) for TF32 support
- PyTorch: 1.7+ for TF32 compatibility and mixed precision
- CUDA: 11.0+ for TF32 operations
- Memory: Sufficient GPU memory for model and batch size

SUITABILITY:
- Best for: Large MoE models, production training, high-throughput scenarios
- Memory: Optimized for GPU memory management with mixed precision
- Performance: ~1.5x speedup over standard FP32 training + memory savings from mixed precision
- Accuracy: ~99.9% accuracy retention with TF32 + mixed precision optimizations

NOT SUITABLE FOR:
- CPU-only environments (use trainer_on_cpu.py instead)
- TPU training (requires different optimizations)
- Older GPUs without TF32 support
- Memory-constrained environments without GPU
"""

import torch
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
from contextlib import nullcontext
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gpt import GPT
from utils.plot_utils import plot_losses
from utils.initialization_utils import apply_gpt2_residual_scaling
from utils.params_util import print_model_info, get_num_params
from utils.training_utils import load_config, get_lr, get_batch, estimate_loss
from utils.memory_utils import memory_optimized_training, MemoryMonitor, optimize_memory_settings
from utils.hellaswag_evaluator import HellaSwagEvalLoader, evaluate_hellaswag, download_hellaswag_data
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
HELLASWAG_SETTING = config["hellaswag"]  # Add HellaSwag settings
WANDB_SETTING = config.get("wandb")  # Add wandb settings
RUN_TYPE = config["run_type"]
torch.manual_seed(TRAINING_SETTING["seed"])

data_dir = DATA_SETTINGS["dataset"]
device = torch.device(DEVICE_SETTING["device_type"])

# Mixed precision configuration
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# -----------------------------------------------------------------------------
print("=" * 50)
print("MIXED PRECISION GPU TRAINING SCRIPT")
print("=" * 50)
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
print(f"Data type: {dtype}")
print(f"Batch size: {TRAINING_SETTING['batch_size']}")
print(f"Model size: {MODEL_SETTING['n_blocks']} layers, {MODEL_SETTING['n_head']} heads, {MODEL_SETTING['n_embd']} embd")
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

data_dir = DATA_SETTINGS["dataset"]

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

print(f"Memory optimization settings: {memory_settings}")

# Initialize HellaSwag evaluation if enabled
hellaswag_eval_loader = None
if HELLASWAG_SETTING["enabled"]:
    print("Setting up HellaSwag evaluation...")
    
    # Check if data exists, download if needed
    hellaswag_data_path = HELLASWAG_SETTING["data_path"]
    if not os.path.exists(hellaswag_data_path):
        if HELLASWAG_SETTING["download_data"]:
            print(f"HellaSwag data not found at {hellaswag_data_path}")
            print("Downloading HellaSwag data...")
            try:
                download_hellaswag_data(os.path.dirname(hellaswag_data_path))
                print(f"HellaSwag data downloaded to {hellaswag_data_path}")
            except Exception as e:
                print(f"Failed to download HellaSwag data: {e}")
                print("Disabling HellaSwag evaluation")
                HELLASWAG_SETTING["enabled"] = False
        else:
            print(f"HellaSwag data not found at {hellaswag_data_path}")
            print("Disabling HellaSwag evaluation")
            HELLASWAG_SETTING["enabled"] = False
    
    if HELLASWAG_SETTING["enabled"]:
        try:
            hellaswag_eval_loader = HellaSwagEvalLoader(
                data_file=hellaswag_data_path,
                batch_size=HELLASWAG_SETTING["batch_size"],
                seq_len=MODEL_SETTING["max_context_length"],
                process_rank=0,
                num_processes=1
            )
            print(f"HellaSwag evaluation ready: {hellaswag_eval_loader.num_examples} examples")
        except Exception as e:
            print(f"Failed to initialize HellaSwag evaluation: {e}")
            print("Disabling HellaSwag evaluation")
            HELLASWAG_SETTING["enabled"] = False
            hellaswag_eval_loader = None

# wandb logging setup
if WANDB_SETTING.get("enabled", False):
    try:
        # Generate run name with timestamp if not specified
        run_name = WANDB_SETTING.get("run_name")
        if run_name:
            run_name = f"{run_name}-{int(time.time())}"
        else:
            run_name = f"mixed-precision-training-{int(time.time())}"
        
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
            'device': DEVICE_SETTING["device_type"],
            'dtype': dtype,
            'accumulation_steps': TRAINING_SETTING.get('accumulation_steps', 1),
            'grad_clip': TRAINING_SETTING.get('grad_clip', 1.0),
            'weight_decay': OPTIMIZER_SETTING.get('weight_decay', 0.1),
            'warmup_iters': LR_SCHEDULE_SETTING.get('warmup_iters', 400),
            'lr_decay_iters': LR_SCHEDULE_SETTING.get('lr_decay_iters', 700),
            'min_lr': LR_SCHEDULE_SETTING.get('min_lr', 0.0001),
            'hellaswag_enabled': HELLASWAG_SETTING.get("enabled", False),
            'hellaswag_eval_interval': HELLASWAG_SETTING.get("eval_interval", 0),
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


def mixed_precision_gpu_trainer(model, optimizer, device, max_iters, eval_interval, log_interval, 
                               grad_clip, iter_num, best_val_loss, eval_only=False, checkpoint_interval=None):
    """
    Train the model with mixed precision and GPU optimizations.
    
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
    hellaswag_accuracies = []  # Track HellaSwag accuracy over time
    tokens_seen = 0
    
    # Create checkpoint directory
    os.makedirs(OUTPUT_SETTINGS["checkpoint_path"], exist_ok=True)
    
    print("Starting mixed precision training...")
    
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
                
                # HellaSwag evaluation
                hellaswag_accuracy = None
                if (HELLASWAG_SETTING["enabled"] and 
                    HELLASWAG_SETTING["eval_interval"] > 0 and 
                    iter_num % HELLASWAG_SETTING["eval_interval"] == 0 and
                    hellaswag_eval_loader is not None):
                    print("Running HellaSwag evaluation...")
                    try:
                        # Use only 2 batches for quick evaluation (similar to loss evaluation)
                        # random_sample=True ensures different questions each time
                        hellaswag_accuracy = evaluate_hellaswag(
                            model, hellaswag_eval_loader, 
                            HELLASWAG_SETTING["batch_size"], 
                            MODEL_SETTING["max_context_length"], 
                            DEVICE_SETTING["device_type"],
                            max_batches=2,  # Just 2 batches
                            random_sample=True  # Random sample each time
                        )
                        print(f"HellaSwag accuracy: {hellaswag_accuracy:.4f}")
                        hellaswag_accuracies.append(hellaswag_accuracy)  # Track accuracy
                    except Exception as e:
                        print(f"HellaSwag evaluation failed: {e}")
                        hellaswag_accuracy = None
                        hellaswag_accuracies.append(None)
                
                # Update memory monitoring
                memory_monitor.update_peak()
                memory_stats = memory_monitor.get_stats()
                
                print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if hellaswag_accuracy is not None:
                    print(f"HellaSwag accuracy: {hellaswag_accuracy:.4f}")
                if device.type == 'cuda':
                    print(f"GPU memory: {memory_stats['current_mb']:.1f}MB (peak: {memory_stats['peak_mb']:.1f}MB)")
                else:
                    print(f"Memory usage: {memory_stats['current_mb']:.1f}MB (peak: {memory_stats['peak_mb']:.1f}MB)")
                
                # wandb logging
                if WANDB_SETTING.get("enabled", False):
                    try:
                        log_data = {
                            "iter": iter_num,
                            "train/loss": losses['train'],
                            "val/loss": losses['val'],
                            "lr": lr,
                            "gpu_memory_mb": memory_stats['current_mb'],
                            "peak_memory_mb": memory_stats['peak_mb'],
                            "tokens_seen": tokens_seen,
                            "dtype": dtype,
                        }
                        if hellaswag_accuracy is not None:
                            log_data["hellaswag/accuracy"] = hellaswag_accuracy
                        wandb.log(log_data)
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")
                
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
                            'dtype': dtype,  # Save mixed precision dtype
                        }
                        if hellaswag_accuracy is not None:
                            checkpoint['hellaswag_accuracy'] = hellaswag_accuracy
                        print(f"Saving checkpoint to {OUTPUT_SETTINGS['checkpoint_path']}")
                        torch.save(checkpoint, os.path.join(OUTPUT_SETTINGS["checkpoint_path"], 'ckpt.pt'))
            
            if iter_num == 0 and eval_only:
                break

            # forward pass with mixed precision
            with ctx:
                logits, loss = model(X, Y)
            
            # backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # gradient clipping
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
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
                    with ctx:  # Use mixed precision context for validation
                        _, val_loss = model(val_X, val_Y)
                    val_lossf = val_loss.item()
                model.train()
                
                memory_stats = memory_monitor.get_stats()
                if device.type == 'cuda':
                    print(f"Iter {iter_num}: train loss {lossf:.4f}, val loss {val_lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, GPU memory {memory_stats['current_mb']:.1f}MB, dtype {dtype}")
                else:
                    print(f"Iter {iter_num}: train loss {lossf:.4f}, val loss {val_lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, memory {memory_stats['current_mb']:.1f}MB, dtype {dtype}")
            
            iter_num += 1

            # termination
            if iter_num > max_iters:
                break

    print("Mixed precision training completed!")
    return train_losses, val_losses, track_tokens_seen, hellaswag_accuracies

# Train the model
train_losses, val_losses, tokens_seen, hellaswag_accuracies = mixed_precision_gpu_trainer(
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

# Run final HellaSwag evaluation if enabled
final_hellaswag_accuracy = None
if (HELLASWAG_SETTING["enabled"] and hellaswag_eval_loader is not None):
    print("Running final HellaSwag evaluation...")
    try:
        # For final evaluation, use more batches for better accuracy
        final_max_batches = HELLASWAG_SETTING.get("final_max_batches", 50)  # More comprehensive final eval
        final_hellaswag_accuracy = evaluate_hellaswag(
            model, hellaswag_eval_loader, 
            HELLASWAG_SETTING["batch_size"], 
            MODEL_SETTING["max_context_length"], 
            DEVICE_SETTING["device_type"],
            max_batches=final_max_batches
        )
        print(f"Final HellaSwag accuracy: {final_hellaswag_accuracy:.4f}")
    except Exception as e:
        print(f"Final HellaSwag evaluation failed: {e}")

final_model_data = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "model_args": MODEL_SETTING,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "tokens_seen": tokens_seen,
    "hellaswag_accuracies": hellaswag_accuracies,
    "memory_stats": final_memory_stats,
    "dtype": dtype,  # Save mixed precision dtype
    "scaler_state": scaler.state_dict(),  # Save scaler state for resuming
}

if final_hellaswag_accuracy is not None:
    final_model_data["hellaswag_accuracy"] = final_hellaswag_accuracy

torch.save(final_model_data, OUTPUT_SETTINGS["model_save_path"])
print(f"Final model saved to {OUTPUT_SETTINGS['model_save_path']}")
print(f"Final memory stats: {final_memory_stats}")
print(f"Training completed with mixed precision dtype: {dtype}")
if final_hellaswag_accuracy is not None:
    print(f"Final HellaSwag accuracy: {final_hellaswag_accuracy:.4f}")

# Finish wandb run
if WANDB_SETTING.get("enabled", False):
    try:
        wandb.finish()
        print("Wandb run finished successfully")
    except Exception as e:
        print(f"Warning: Failed to finish wandb run: {e}")
