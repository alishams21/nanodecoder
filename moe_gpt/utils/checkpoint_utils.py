import os
import torch
from typing import Dict, Any, Tuple, Optional


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
        
    Returns:
        Loaded checkpoint dictionary
    """
    ckpt_path = os.path.join(checkpoint_path, 'ckpt.pt')
    return torch.load(ckpt_path, map_location=device)


def update_model_settings_from_checkpoint(checkpoint: Dict[str, Any], model_setting: Dict[str, Any]) -> None:
    """
    Update model settings from checkpoint model args.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        model_setting: Model configuration dictionary to update
    """
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_blocks', 'n_head', 'n_embd', 'max_context_length', 'bias', 'vocab_size']:
        model_setting[k] = checkpoint_model_args[k]


def resume_from_checkpoint(checkpoint_path: str, model_setting: Dict[str, Any], 
                         device: torch.device, master_process: bool = True) -> Tuple[Dict[str, Any], int, float]:
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        model_setting: Model configuration dictionary to update
        device: Device to load the checkpoint on
        master_process: Whether this is the master process (for distributed training)
        
    Returns:
        Tuple of (checkpoint_dict, iter_num, best_val_loss)
    """
    if master_process:
        print(f"Resuming training from {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path, device)
    update_model_settings_from_checkpoint(checkpoint, model_setting)
    
    return checkpoint, checkpoint['iter_num'], checkpoint['best_val_loss']


def load_optimizer_from_checkpoint(optimizer, checkpoint: Dict[str, Any]) -> None:
    """
    Load optimizer state from checkpoint if available.
    
    Args:
        optimizer: Optimizer to load state into
        checkpoint: Loaded checkpoint dictionary
    """
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])


def create_checkpoint(model, optimizer, iter_num: int, model_args: Dict[str, Any], 
                     best_val_loss: float) -> Dict[str, Any]:
    """
    Create a checkpoint dictionary.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        iter_num: Current iteration number
        model_args: Model configuration arguments
        best_val_loss: Best validation loss so far
        
    Returns:
        Checkpoint dictionary
    """
    return {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }


def save_checkpoint(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """
    Save checkpoint to file.
    
    Args:
        checkpoint: Checkpoint dictionary to save
        checkpoint_path: Path to save the checkpoint
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_path, 'ckpt.pt')
    torch.save(checkpoint, ckpt_path)
