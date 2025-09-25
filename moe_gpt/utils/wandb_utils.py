"""
Weights & Biases (wandb) logging utilities for MoE GPT training.

This module provides centralized wandb logging functionality that can be used
across different training scripts to maintain consistent logging.
"""

import time
from typing import Dict, Any, Optional
import wandb


def create_wandb_config(model_setting: Dict[str, Any], training_setting: Dict[str, Any], 
                       optimizer_setting: Dict[str, Any], lr_schedule_setting: Dict[str, Any],
                       device_setting: Dict[str, Any], hellaswag_setting: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create wandb configuration dictionary from all settings.
    
    Args:
        model_setting: Model configuration
        training_setting: Training configuration
        optimizer_setting: Optimizer configuration
        lr_schedule_setting: Learning rate schedule configuration
        device_setting: Device configuration
        hellaswag_setting: HellaSwag evaluation configuration
        
    Returns:
        Wandb configuration dictionary
    """
    return {
        'n_layer': model_setting['n_blocks'],
        'n_head': model_setting['n_head'], 
        'n_embd': model_setting['n_embd'],
        'n_exp': model_setting.get('n_exp', 0),
        'top_k': model_setting.get('top_k', 1),
        'use_aux_loss': model_setting.get('use_aux_loss', False),
        'use_router_z_loss': model_setting.get('use_router_z_loss', False),
        'learning_rate': optimizer_setting['learning_rate'],
        'batch_size': training_setting['batch_size'],
        'block_size': model_setting['max_context_length'],
        'device': device_setting["device_type"],
        'accumulation_steps': training_setting.get('accumulation_steps', 1),
        'grad_clip': training_setting.get('grad_clip', 1.0),
        'weight_decay': optimizer_setting.get('weight_decay', 0.1),
        'warmup_iters': lr_schedule_setting.get('warmup_iters', 400),
        'lr_decay_iters': lr_schedule_setting.get('lr_decay_iters', 700),
        'min_lr': lr_schedule_setting.get('min_lr', 0.0001),
        'hellaswag_enabled': hellaswag_setting.get("enabled", False),
        'hellaswag_eval_interval': hellaswag_setting.get("eval_interval", 0),
    }


def generate_run_name(wandb_setting: Dict[str, Any], default_prefix: str = "training") -> str:
    """
    Generate wandb run name with timestamp.
    
    Args:
        wandb_setting: Wandb configuration
        default_prefix: Default prefix if no run name is specified
        
    Returns:
        Generated run name
    """
    run_name = wandb_setting.get("run_name")
    if run_name:
        return f"{run_name}-{int(time.time())}"
    else:
        return f"{default_prefix}-{int(time.time())}"


def initialize_wandb(wandb_setting: Dict[str, Any], wandb_config: Dict[str, Any], 
                    run_name: str) -> bool:
    """
    Initialize wandb logging.
    
    Args:
        wandb_setting: Wandb configuration
        wandb_config: Configuration dictionary to log
        run_name: Name for the wandb run
        
    Returns:
        True if initialization successful, False otherwise
    """
    if not wandb_setting.get("enabled", False):
        return False
    
    try:
        wandb.init(
            project=wandb_setting.get("project"),
            name=run_name,
            config=wandb_config
        )
        print(f"Wandb initialized: {run_name}")
        return True
    except ImportError:
        print("Warning: wandb not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return False


def setup_wandb_logging(wandb_setting: Dict[str, Any], model_setting: Dict[str, Any],
                       training_setting: Dict[str, Any], optimizer_setting: Dict[str, Any],
                       lr_schedule_setting: Dict[str, Any], device_setting: Dict[str, Any],
                       hellaswag_setting: Dict[str, Any], default_prefix: str = "training") -> bool:
    """
    Complete wandb setup including configuration creation and initialization.
    
    Args:
        wandb_setting: Wandb configuration
        model_setting: Model configuration
        training_setting: Training configuration
        optimizer_setting: Optimizer configuration
        lr_schedule_setting: Learning rate schedule configuration
        device_setting: Device configuration
        hellaswag_setting: HellaSwag evaluation configuration
        default_prefix: Default prefix for run name
        
    Returns:
        True if setup successful, False otherwise
    """
    if not wandb_setting.get("enabled", False):
        return False
    
    # Create wandb configuration
    wandb_config = create_wandb_config(
        model_setting, training_setting, optimizer_setting,
        lr_schedule_setting, device_setting, hellaswag_setting
    )
    
    # Generate run name
    run_name = generate_run_name(wandb_setting, default_prefix)
    
    # Initialize wandb
    return initialize_wandb(wandb_setting, wandb_config, run_name)


def log_training_metrics(iter_num: int, train_loss: float, val_loss: float, 
                        learning_rate: float, hellaswag_accuracy: Optional[float] = None,
                        wandb_setting: Optional[Dict[str, Any]] = None) -> None:
    """
    Log training metrics to wandb.
    
    Args:
        iter_num: Current iteration number
        train_loss: Training loss
        val_loss: Validation loss
        learning_rate: Current learning rate
        hellaswag_accuracy: HellaSwag accuracy (optional)
        wandb_setting: Wandb configuration (optional, for checking if enabled)
    """
    if wandb_setting and not wandb_setting.get("enabled", False):
        return
    
    try:
        log_data = {
            "iter": iter_num,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "learning_rate": learning_rate,
        }
        
        if hellaswag_accuracy is not None:
            log_data["hellaswag/accuracy"] = hellaswag_accuracy
        
        wandb.log(log_data)
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")


def finish_wandb_run(wandb_setting: Optional[Dict[str, Any]] = None) -> None:
    """
    Finish wandb run.
    
    Args:
        wandb_setting: Wandb configuration (optional, for checking if enabled)
    """
    if wandb_setting and not wandb_setting.get("enabled", False):
        return
    
    try:
        wandb.finish()
        print("Wandb run finished successfully")
    except Exception as e:
        print(f"Warning: Failed to finish wandb run: {e}")


def log_model_info(model, wandb_setting: Optional[Dict[str, Any]] = None) -> None:
    """
    Log model information to wandb.
    
    Args:
        model: Model to log information about
        wandb_setting: Wandb configuration (optional, for checking if enabled)
    """
    if wandb_setting and not wandb_setting.get("enabled", False):
        return
    
    try:
        # Log model parameters count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/total_parameters_millions": total_params / 1e6,
        })
    except Exception as e:
        print(f"Warning: Failed to log model info to wandb: {e}")


def log_checkpoint_info(checkpoint_path: str, iter_num: int, 
                       wandb_setting: Optional[Dict[str, Any]] = None) -> None:
    """
    Log checkpoint information to wandb.
    
    Args:
        checkpoint_path: Path to the checkpoint
        iter_num: Iteration number when checkpoint was saved
        wandb_setting: Wandb configuration (optional, for checking if enabled)
    """
    if wandb_setting and not wandb_setting.get("enabled", False):
        return
    
    try:
        wandb.log({
            "checkpoint/path": checkpoint_path,
            "checkpoint/iter": iter_num,
        })
    except Exception as e:
        print(f"Warning: Failed to log checkpoint info to wandb: {e}")
