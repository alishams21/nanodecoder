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
from utils.hellaswag_utils import setup_hellaswag_evaluation, run_hellaswag_evaluation, should_run_hellaswag_evaluation
import wandb

from utils.vocab_utils import load_and_update_vocab_size
from utils.checkpoint_utils import resume_from_checkpoint, load_optimizer_from_checkpoint
from utils.wandb_utils import setup_wandb_logging, log_training_metrics, finish_wandb_run, log_model_info
from utils.ddp_utils import setup_distributed_training, get_ddp_info, print_ddp_info, cleanup_ddp

from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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
COMPILE_SETTING = config.get("compile")  # Default to enabled if not specified
torch.manual_seed(TRAINING_SETTING["seed"])

data_dir = DATA_SETTINGS["dataset"]
device = torch.device(DEVICE_SETTING["device_type"])

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=DEVICE_SETTING["backend"])
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    gradient_accumulation_steps = TRAINING_SETTING["gradient_accumulation_steps"]
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    gradient_accumulation_steps = TRAINING_SETTING["gradient_accumulation_steps"]


tokens_per_iter = gradient_accumulation_steps * ddp_world_size * TRAINING_SETTING["batch_size"] * MODEL_SETTING["max_context_length"]
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    # Create checkpoint directory
    os.makedirs(OUTPUT_SETTINGS["checkpoint_path"], exist_ok=True)
    

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device = torch.device(DEVICE_SETTING["device_type"])

# Mixed precision configuration
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# Mixed precision context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if DEVICE_SETTING["device_type"] == 'cpu' else torch.amp.autocast(device_type=DEVICE_SETTING["device_type"], dtype=ptdtype)
 
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
if master_process:
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print_model_info(model, non_embedding=True)
    print(f"Model created with {get_num_params(model)/1e6:.2f}M parameters")
    print(f"Wandb logging: {WANDB_SETTING.get('enabled', False)}")
    print("=" * 50)


if DEVICE_SETTING["device_type"] == 'cuda' and dtype == 'float16':
    scaler = torch.cuda.amp.GradScaler(DEVICE_SETTING["device_type"], enabled=True)
else:
    scaler = torch.cuda.amp.GradScaler(DEVICE_SETTING["device_type"], enabled=False)  # No-op scaler for CUDA

# # Compile the model for faster training (PyTorch 2.0+)
# if COMPILE_SETTING.get("enabled", True):
#     print("Compiling the model... (takes a ~minute)")
#     unoptimized_model = model
#     model = torch.compile(model)  # requires PyTorch 2.0
#     print("Model compilation completed!")

if ddp:
    # Wrap model in DDP
    model = DDP(model, device_ids=[ddp_local_rank])

model.to(device)


# Load monitor
memory_monitor = MemoryMonitor(device)

# Load HellaSwag evaluation
hellaswag_eval_loader, HELLASWAG_SETTING = setup_hellaswag_evaluation(HELLASWAG_SETTING, MODEL_SETTING)

# wandb logging setup
wandb_enabled = setup_wandb_logging(
    WANDB_SETTING, MODEL_SETTING, TRAINING_SETTING, 
    OPTIMIZER_SETTING, LR_SCHEDULE_SETTING, DEVICE_SETTING, 
    HELLASWAG_SETTING, f"training_{DEVICE_SETTING["device_type"]}"
)

if wandb_enabled:
    log_model_info(model, WANDB_SETTING)
                   
#auto-optimize settings
memory_settings = {}
if MEMORY_SETTING["auto_optimize"]:
    auto_settings = optimize_memory_settings(device, TRAINING_SETTING["batch_size"], 
                                           get_num_params(model) * 4 / 1024 / 1024)
    memory_settings.update(auto_settings)
else:
    memory_settings.update(MEMORY_SETTING)
    
raw_model = model.module if ddp else model # unwrap DDP container if needed
# init training state
def trainer(model, optimizer, device, max_iters, eval_interval, log_interval, 
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
    local_iter_num = 0 # number of iterations in the lifetime of this process
    running_mfu = -1.0
    

    
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
            if iter_num % eval_interval == 0 and master_process:
                losses = estimate_loss(model, get_batch_wrapper, TRAINING_SETTING)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                track_tokens_seen.append(tokens_seen)
                
            if iter_num % checkpoint_interval and master_process == 0:
                best_val_loss = losses['val']
                if iter_num > 0:
                    current_memory_stats = memory_monitor.get_stats()
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': MODEL_SETTING,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'memory_stats': current_memory_stats,
                        'hellaswag_accuracy': hellaswag_accuracies[-1] if hellaswag_accuracies else 0.0,
                    }
                    torch.save(checkpoint, os.path.join(OUTPUT_SETTINGS["checkpoint_path"], 'ckpt.pt'))
            
            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # get next batch using memory-optimized loader
                X, Y = async_loader.get_batch()
                tokens_seen += X.numel()
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            
            # clip the gradient and calculate gradient norm
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_SETTING["grad_clip"])

            
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)    

            # logging
            if iter_num % log_interval == 0 and master_process:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                # training loss
                lossf = loss.item() * gradient_accumulation_steps

                # memory stats
                memory_monitor.update_peak()
                memory_stats = memory_monitor.get_stats()

                # MFU
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(
                        TRAINING_SETTING["batch_size"] * gradient_accumulation_steps, dt
                    )
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

                # only evaluate validation loss at eval_interval
                val_lossf = None
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
                        max_batches=2
                    )
                    hellaswag_accuracies.append(hellaswag_accuracy)

                # fallback if no evaluation this step
                if val_lossf is None:
                    val_lossf = float('nan')
                    hellaswag_accuracy = hellaswag_accuracies[-1] if hellaswag_accuracies else float('nan')

                # combined logging
                print(
                    f"Iter {iter_num}: "
                    f"train loss {lossf:.4f}, val loss {val_lossf:.4f}, "
                    f"grad_norm {grad_norm:.6f}, "
                    f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, "
                    f"memory {memory_stats['current_mb']:.1f}MB, "
                    f"hellaswag_accuracy {hellaswag_accuracy}"
                )

                # wandb logging
                if WANDB_SETTING.get("enabled", False):
                    log_training_metrics(
                        iter_num,
                        losses['train'],
                        losses['val'] if iter_num % eval_interval == 0 else None,
                        get_lr(iter_num, LR_SCHEDULE_SETTING, OPTIMIZER_SETTING),
                        hellaswag_accuracy,
                        WANDB_SETTING,
                        grad_norm=grad_norm
                    )

        
            iter_num += 1
            local_iter_num += 1                   
            

            # termination
            if iter_num > max_iters:
                break

    print("Training completed!")
    return train_losses, val_losses, track_tokens_seen, hellaswag_accuracies

# Train the model
train_losses, val_losses, tokens_seen, hellaswag_accuracies = trainer(
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

# At the end of the file, add cleanup
cleanup_ddp()
