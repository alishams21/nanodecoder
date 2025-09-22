"""
Memory Management Utilities for GPT Training

This module provides memory optimization utilities for both CPU and GPU training:
- Batch prefetching for improved data loading efficiency
- Memory pooling for reduced allocation overhead
- Asynchronous data loading for better pipeline efficiency
"""

import torch
import threading
import queue
import time
from typing import Callable, Tuple, Optional, Any
from contextlib import contextmanager


class MemoryPool:
    """
    Memory pool for reusing tensor memory to reduce allocation overhead.
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, max_size: int = 4):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.max_size = max_size
        self.pool = []
        self.lock = threading.Lock()
    
    def get_tensor(self) -> torch.Tensor:
        """Get a tensor from the pool or create a new one."""
        with self.lock:
            if self.pool:
                tensor = self.pool.pop()
                # Clear the tensor to avoid data leakage
                tensor.zero_()
                return tensor
            else:
                return torch.empty(self.shape, dtype=self.dtype, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse."""
        if tensor.shape == self.shape and tensor.dtype == self.dtype and tensor.device == self.device:
            with self.lock:
                if len(self.pool) < self.max_size:
                    self.pool.append(tensor.detach())
    
    def clear(self) -> None:
        """Clear the memory pool."""
        with self.lock:
            self.pool.clear()


class BatchPrefetcher:
    """
    Batch prefetcher that loads the next batch while the current batch is being processed.
    """
    
    def __init__(self, get_batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]], 
                 buffer_size: int = 2, device: torch.device = None):
        self.get_batch_fn = get_batch_fn
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.thread = None
        self.stop_event = threading.Event()
        self._start_prefetching()
    
    def _start_prefetching(self) -> None:
        """Start the prefetching thread."""
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
    
    def _prefetch_worker(self) -> None:
        """Worker thread that continuously prefetches batches."""
        while not self.stop_event.is_set():
            try:
                batch = self.get_batch_fn('train')
                if self.device is not None:
                    batch = (batch[0].to(self.device, non_blocking=True), 
                           batch[1].to(self.device, non_blocking=True))
                self.buffer.put(batch, timeout=1.0)
            except queue.Full:
                # Buffer is full, skip this batch
                continue
            except Exception as e:
                print(f"Warning: Error in prefetch worker: {e}")
                time.sleep(0.1)
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the next prefetched batch."""
        try:
            return self.buffer.get(timeout=5.0)
        except queue.Empty:
            # Fallback to direct loading if prefetching fails
            print("Warning: Prefetch buffer empty, falling back to direct loading")
            batch = self.get_batch_fn('train')
            if self.device is not None:
                batch = (batch[0].to(self.device, non_blocking=True), 
                       batch[1].to(self.device, non_blocking=True))
            return batch
    
    def stop(self) -> None:
        """Stop the prefetching thread."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class AsyncDataLoader:
    """
    Asynchronous data loader with memory pooling and prefetching.
    """
    
    def __init__(self, get_batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]], 
                 device: torch.device, buffer_size: int = 2, enable_memory_pool: bool = True):
        self.get_batch_fn = get_batch_fn
        self.device = device
        self.buffer_size = buffer_size
        self.enable_memory_pool = enable_memory_pool
        
        # Initialize memory pools for X and Y tensors
        self.memory_pools = {}
        if enable_memory_pool:
            self.memory_pools = {
                'X': MemoryPool((0, 0), torch.long, device),
                'Y': MemoryPool((0, 0), torch.long, device)
            }
        
        # Initialize prefetcher
        self.prefetcher = BatchPrefetcher(self._load_batch_with_pooling, buffer_size, device)
    
    def _load_batch_with_pooling(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load batch with memory pooling."""
        X, Y = self.get_batch_fn(split)
        
        if self.enable_memory_pool:
            # Try to reuse memory from pools
            X_pooled = self.memory_pools['X'].get_tensor()
            Y_pooled = self.memory_pools['Y'].get_tensor()
            
            # Resize pooled tensors if needed
            if X_pooled.shape != X.shape:
                X_pooled = torch.empty_like(X)
            if Y_pooled.shape != Y.shape:
                Y_pooled = torch.empty_like(Y)
            
            # Copy data to pooled tensors
            X_pooled.copy_(X)
            Y_pooled.copy_(Y)
            
            return X_pooled, Y_pooled
        
        return X, Y
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch."""
        return self.prefetcher.get_batch()
    
    def return_batch(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Return batch tensors to memory pools."""
        if self.enable_memory_pool:
            self.memory_pools['X'].return_tensor(X)
            self.memory_pools['Y'].return_tensor(Y)
    
    def stop(self) -> None:
        """Stop the async loader."""
        self.prefetcher.stop()
        for pool in self.memory_pools.values():
            pool.clear()


@contextmanager
def memory_optimized_training(get_batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]], 
                            device: torch.device, buffer_size: int = 2, 
                            enable_memory_pool: bool = True):
    """
    Context manager for memory-optimized training.
    
    Args:
        get_batch_fn: Function to get batches
        device: Device to use
        buffer_size: Size of prefetch buffer
        enable_memory_pool: Whether to enable memory pooling
    
    Yields:
        AsyncDataLoader: The memory-optimized data loader
    """
    loader = AsyncDataLoader(get_batch_fn, device, buffer_size, enable_memory_pool)
    try:
        yield loader
    finally:
        loader.stop()


class MemoryMonitor:
    """
    Memory usage monitor for tracking memory consumption during training.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        self.memory_history = []
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def update_peak(self) -> None:
        """Update peak memory usage."""
        current = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current)
        self.memory_history.append(current)
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        current = self.get_memory_usage()
        return {
            'current_mb': current,
            'peak_mb': self.peak_memory,
            'avg_mb': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        }


def optimize_memory_settings(device: torch.device, batch_size: int, model_size_mb: float) -> dict:
    """
    Get optimized memory settings based on device and model size.
    
    Args:
        device: Device to optimize for
        batch_size: Training batch size
        model_size_mb: Model size in MB
    
    Returns:
        dict: Optimized memory settings
    """
    settings = {
        'buffer_size': 2,
        'enable_memory_pool': True,
        'max_pool_size': 4
    }
    
    if device.type == 'cuda':
        # GPU-specific optimizations
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
        if gpu_memory > 8000:  # High-end GPU
            settings['buffer_size'] = 4
            settings['max_pool_size'] = 8
        elif gpu_memory > 4000:  # Mid-range GPU
            settings['buffer_size'] = 3
            settings['max_pool_size'] = 6
    else:
        # CPU-specific optimizations
        import psutil
        total_memory = psutil.virtual_memory().total / 1024 / 1024
        if total_memory > 16000:  # 16GB+ RAM
            settings['buffer_size'] = 4
            settings['max_pool_size'] = 8
        elif total_memory > 8000:  # 8GB+ RAM
            settings['buffer_size'] = 3
            settings['max_pool_size'] = 6
    
    return settings
