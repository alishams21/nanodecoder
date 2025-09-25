"""
Distributed Data Parallel (DDP) utilities for multi-GPU training.

This module provides utilities for setting up and managing distributed training
across multiple GPUs using PyTorch's DistributedDataParallel.
"""

import os
import sys
import subprocess
import torch
from torch.distributed import init_process_group
from typing import Dict, Any, Tuple


def check_ddp_environment() -> bool:
    """
    Check if we're running in a distributed environment.
    
    Returns:
        True if running in DDP mode, False otherwise
    """
    return int(os.environ.get('RANK', -1)) != -1


def auto_launch_distributed_training(device_setting: Dict[str, Any], script_path: str) -> None:
    """
    Auto-launch distributed training if multiple GPUs are requested.
    
    Args:
        device_setting: Device configuration dictionary
        script_path: Path to the training script
    """
    ddp = check_ddp_environment()
    nproc_per_node = device_setting.get("nproc_per_node", 1)
    
    if not ddp and nproc_per_node > 1:
        available_gpus = torch.cuda.device_count()
        if available_gpus >= nproc_per_node:
            print(f"Auto-launching distributed training with {nproc_per_node} GPUs...")
            
            # Launch torchrun with the configured number of processes
            cmd = [
                sys.executable, "-m", "torch.distributed.launch",
                f"--nproc_per_node={nproc_per_node}",
                "--use_env",
                script_path
            ]
            
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)
            sys.exit(0)
        else:
            print(f"Warning: Requested {nproc_per_node} GPUs but only {available_gpus} available. Falling back to single GPU training.")


def setup_ddp_environment(device_setting: Dict[str, Any]) -> Tuple[bool, int, int, int, torch.device, bool, int, int]:
    """
    Setup distributed training environment.
    
    Args:
        device_setting: Device configuration dictionary
        
    Returns:
        Tuple of (ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset, nproc_per_node)
    """
    ddp = check_ddp_environment()
    nproc_per_node = device_setting.get("nproc_per_node", 1)
    
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{ddp_local_rank}')
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = nproc_per_node
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ddp_rank = 0
        ddp_local_rank = 0
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset, nproc_per_node


def setup_distributed_training(device_setting: Dict[str, Any], script_path: str) -> Tuple[bool, int, int, int, torch.device, bool, int, int]:
    """
    Complete distributed training setup including auto-launch and environment setup.
    
    Args:
        device_setting: Device configuration dictionary
        script_path: Path to the training script
        
    Returns:
        Tuple of (ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset, nproc_per_node)
    """
    # Auto-launch if needed
    auto_launch_distributed_training(device_setting, script_path)
    
    # Setup DDP environment
    return setup_ddp_environment(device_setting)


def get_ddp_info() -> Dict[str, Any]:
    """
    Get current DDP information.
    
    Returns:
        Dictionary with DDP information
    """
    ddp = check_ddp_environment()
    
    if ddp:
        return {
            'ddp': True,
            'rank': int(os.environ.get('RANK', 0)),
            'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
            'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
            'master_port': os.environ.get('MASTER_PORT', '12355')
        }
    else:
        return {
            'ddp': False,
            'rank': 0,
            'local_rank': 0,
            'world_size': 1,
            'master_addr': 'localhost',
            'master_port': '12355'
        }


def print_ddp_info(ddp_info: Dict[str, Any]) -> None:
    """
    Print DDP information.
    
    Args:
        ddp_info: DDP information dictionary
    """
    print("=" * 50)
    print("DISTRIBUTED DATA PARALLEL (DDP) INFORMATION")
    print("=" * 50)
    if ddp_info['ddp']:
        print(f"DDP enabled: {ddp_info['ddp']}")
        print(f"Rank: {ddp_info['rank']}/{ddp_info['world_size']}")
        print(f"Local rank: {ddp_info['local_rank']}")
        print(f"Master address: {ddp_info['master_addr']}")
        print(f"Master port: {ddp_info['master_port']}")
    else:
        print(f"DDP enabled: {ddp_info['ddp']}")
        print("Running in single-process mode")
    print("=" * 50)


def cleanup_ddp() -> None:
    """
    Cleanup distributed training environment.
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
