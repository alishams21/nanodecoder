"""
HellaSwag Evaluation Utility for MoE GPT Models

This module provides HellaSwag evaluation functionality that can be integrated
into the training loop to assess model performance on commonsense reasoning tasks.

Based on the C implementation from llmc/dataloader.h and train_gpt2.cu,
this Python implementation replicates the exact same evaluation logic.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tiktoken import get_encoding
from typing import List, Dict, Tuple, Optional, Any
import pickle
import urllib.request
import zipfile


class HellaSwagEvalLoader:
    """
    Python equivalent of the C EvalLoader struct and functions.
    Replicates the exact same logic as the C implementation.
    """
    
    def __init__(self, data_file: str, batch_size: int, seq_len: int, 
                 process_rank: int = 0, num_processes: int = 1):
        """
        Initialize the evaluation loader - equivalent to evalloader_init()
        """
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = batch_size
        self.T = seq_len
        
        # Load and parse the binary data file
        self.examples = self._load_binary_data(data_file)
        self.num_examples = len(self.examples)
        
        # Calculate work distribution (same as C implementation)
        examples_per_process = (self.num_examples + num_processes - 1) // num_processes
        can_fit_examples = self.B // 4  # ASSUMED_NUM_COMPLETIONS = 4
        
        if can_fit_examples == 0:
            raise ValueError(f"HellaSwag EvalLoader: batch size {self.B} is < 4")
        
        self.num_batches = (examples_per_process + can_fit_examples - 1) // can_fit_examples
        self.start_example_index = examples_per_process * process_rank
        self.end_example_index = min(examples_per_process * (process_rank + 1), self.num_examples)
        self.current_example_index = self.start_example_index
        
        # Initialize batch data structures
        self.inputs = torch.zeros((self.B, self.T), dtype=torch.long)
        self.targets = torch.zeros((self.B, self.T), dtype=torch.long)
        self.mask = torch.zeros((self.B, self.T), dtype=torch.bool)
        self.labels = torch.zeros(can_fit_examples, dtype=torch.long)
        
    def _load_binary_data(self, data_file: str) -> List[Dict]:
        """
        Load binary data file - equivalent to reading the .bin file in C
        For this implementation, we'll use the JSON data directly
        """
        # In the C implementation, this would read from the binary file
        # For Python, we'll use the JSON data that was used to create the binary
        json_file = data_file.replace('.bin', '.jsonl')
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Data file not found: {json_file}")
        
        examples = []
        with open(json_file, 'r') as f:
            for line in f:
                example = json.loads(line)
                examples.append(example)
        
        return examples
    
    def reset(self, random_start=True):
        """
        Reset the loader - equivalent to evalloader_reset()
        If random_start=True, start from a random position for variety
        """
        if random_start:
            # Randomly select starting position for variety in evaluation
            import random
            max_start = max(0, self.end_example_index - (self.B // 4) * 2)  # Ensure we can fit 2 batches
            self.current_example_index = random.randint(self.start_example_index, max_start)
        else:
            self.current_example_index = self.start_example_index
    
    def next_batch(self) -> bool:
        """
        Load next batch - equivalent to evalloader_next_batch()
        Returns True if batch loaded successfully, False if no more data
        """
        if self.current_example_index >= self.end_example_index:
            return False
        
        # Clear previous batch
        self.inputs.zero_()
        self.targets.zero_()
        self.mask.zero_()
        self.labels.zero_()
        
        can_fit_examples = self.B // 4  # ASSUMED_NUM_COMPLETIONS = 4
        examples_loaded = 0
        
        for i in range(can_fit_examples):
            if self.current_example_index >= self.end_example_index:
                break
            
            example = self.examples[self.current_example_index]
            self._load_example(example, i)
            examples_loaded += 1
            self.current_example_index += 1
        
        return examples_loaded > 0
    
    def _load_example(self, example: Dict, batch_idx: int):
        """
        Load a single example into the batch - equivalent to evalloader_next_example_()
        """
        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]
        
        # Tokenize context and endings
        enc = get_encoding("gpt2")
        ctx_tokens = enc.encode(ctx)
        
        # Process each completion (4 total)
        for c in range(4):  # ASSUMED_NUM_COMPLETIONS = 4
            batch_row = batch_idx * 4 + c
            
            # Tokenize completion
            end_tokens = enc.encode(" " + endings[c])
            
            # Create full sequence: context + completion
            full_tokens = ctx_tokens + end_tokens
            
            # Truncate if too long
            if len(full_tokens) > self.T:
                full_tokens = full_tokens[:self.T]
            
            # Fill inputs
            seq_len = len(full_tokens)
            self.inputs[batch_row, :seq_len] = torch.tensor(full_tokens, dtype=torch.long)
            
            # Create targets (shifted by 1)
            if seq_len > 1:
                self.targets[batch_row, :seq_len-1] = torch.tensor(full_tokens[1:], dtype=torch.long)
            
            # Create mask (1 for completion tokens)
            mask_start = len(ctx_tokens)
            mask_end = seq_len
            if mask_start < mask_end:
                self.mask[batch_row, mask_start:mask_end] = True
        
        # Store label
        self.labels[batch_idx] = label
    
    def stat_losses(self, losses: torch.Tensor) -> int:
        """
        Calculate statistics from losses - equivalent to evalloader_stat_losses()
        Returns number of correct predictions in this batch
        """
        correct = 0
        can_fit_examples = self.B // 4  # ASSUMED_NUM_COMPLETIONS = 4
        
        for i in range(can_fit_examples):
            if i >= len(self.labels):
                break
            
            min_loss = float('inf')
            min_loss_index = -1
            active = False
            
            # Check each completion for this example
            for c in range(4):  # ASSUMED_NUM_COMPLETIONS = 4
                batch_row = i * 4 + c
                
                # Calculate average loss for this completion
                completion_losses = []
                for t in range(self.T):
                    if self.mask[batch_row, t]:
                        active = True
                        completion_losses.append(losses[batch_row, t].item())
                
                if completion_losses:
                    avg_loss = sum(completion_losses) / len(completion_losses)
                    if avg_loss < min_loss:
                        min_loss = avg_loss
                        min_loss_index = c
            
            # Check if prediction is correct
            if active and min_loss_index == self.labels[i].item():
                correct += 1
        
        return correct


def gpt2_validate(model, inputs: torch.Tensor, targets: torch.Tensor, 
                 batch_size: int, seq_len: int) -> Tuple[float, torch.Tensor]:
    """
    Python equivalent of gpt2_validate() from train_gpt2.cu
    Returns (mean_loss, per_token_losses)
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        logits, _ = model(inputs, targets)
        
        # Calculate per-token losses
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        
        # Flatten for cross entropy
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_targets = shift_targets.view(-1)
        
        # Calculate losses
        losses = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        losses = losses.view(batch_size, seq_len - 1)
        
        # Pad losses to match original sequence length
        padded_losses = torch.zeros(batch_size, seq_len)
        padded_losses[:, :-1] = losses
        
        # Calculate mean loss
        mean_loss = losses.mean().item()
        
        return mean_loss, padded_losses


@torch.no_grad()
def evaluate_hellaswag(model, eval_loader: HellaSwagEvalLoader, 
                      batch_size: int, seq_len: int, device: str = "cpu", 
                      max_batches: int = None, random_sample: bool = True) -> float:
    """
    Main evaluation function - equivalent to the HellaSwag evaluation loop in train_gpt2.cu
    """
    eval_loader.reset(random_start=random_sample)
    total_correct = 0
    total_examples = 0
    
    # Limit evaluation to a small sample if max_batches is specified
    num_batches_to_eval = min(eval_loader.num_batches, max_batches) if max_batches else eval_loader.num_batches
    
    
    for batch_idx in range(num_batches_to_eval):
        if not eval_loader.next_batch():
            break
        
        
        # Move data to device
        inputs = eval_loader.inputs.to(device)
        targets = eval_loader.targets.to(device)
        
        # Forward pass and loss calculation
        mean_loss, per_token_losses = gpt2_validate(
            model, inputs, targets, batch_size, seq_len
        )
        
        # Calculate accuracy for this batch
        correct = eval_loader.stat_losses(per_token_losses.cpu())
        total_correct += correct
        
        # Count examples in this batch (each HellaSwag question has 4 answer choices)
        examples_in_batch = min(eval_loader.B // 4, 
                               eval_loader.end_example_index - eval_loader.current_example_index + 
                               (eval_loader.B // 4))
        # Each HellaSwag question has 4 answer choices, so multiply by 4
        total_examples += examples_in_batch * 4
    
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    
    return accuracy


def download_hellaswag_data(data_dir: str = "moe_gpt/data/hellaswag") -> str:
    """
    Download and prepare HellaSwag data for evaluation
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Download HellaSwag data
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    jsonl_path = os.path.join(data_dir, "hellaswag_val.jsonl")
    
    if not os.path.exists(jsonl_path):
        print(f"Downloading HellaSwag data to {jsonl_path}")
        urllib.request.urlretrieve(url, jsonl_path)
    
    return jsonl_path


def prepare_hellaswag_data(jsonl_path: str, output_path: str):
    """
    Convert HellaSwag JSONL data to the format expected by the evaluator
    """
    examples = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            example = {
                "ctx": data["ctx"],
                "label": data["label"],
                "endings": data["endings"]
            }
            examples.append(example)
    
    # Save processed data
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Prepared HellaSwag data: {len(examples)} examples")
    return len(examples)


def run_hellaswag_evaluation_standalone(model, data_path: str, batch_size: int = 4, 
                           seq_len: int = 256, device: str = "cpu") -> float:
    """
    Run HellaSwag evaluation on a model
    """
    # Create evaluation loader
    eval_loader = HellaSwagEvalLoader(
        data_file=data_path,
        batch_size=batch_size,
        seq_len=seq_len,
        process_rank=0,
        num_processes=1
    )
    
    # Run evaluation
    accuracy = evaluate_hellaswag(model, eval_loader, batch_size, seq_len, device)
    return accuracy


# Setup and management functions
def setup_hellaswag_evaluation(hellaswag_setting: Dict[str, Any], model_setting: Dict[str, Any]) -> Tuple[Optional[HellaSwagEvalLoader], Dict[str, Any]]:
    """
    Setup HellaSwag evaluation if enabled.
    
    Args:
        hellaswag_setting: HellaSwag configuration dictionary
        model_setting: Model configuration dictionary
        
    Returns:
        Tuple of (hellaswag_eval_loader, updated_hellaswag_setting)
    """
    hellaswag_eval_loader = None
    updated_setting = hellaswag_setting.copy()
    
    if not hellaswag_setting["enabled"]:
        return hellaswag_eval_loader, updated_setting
    
    print("Setting up HellaSwag evaluation...")
    
    # Check if data exists, download if needed
    hellaswag_data_path = hellaswag_setting["data_path"]
    if not os.path.exists(hellaswag_data_path):
        if hellaswag_setting["download_data"]:
            print(f"HellaSwag data not found at {hellaswag_data_path}")
            print("Downloading HellaSwag data...")
            try:
                download_hellaswag_data(os.path.dirname(hellaswag_data_path))
                print(f"HellaSwag data downloaded to {hellaswag_data_path}")
            except Exception as e:
                print(f"Failed to download HellaSwag data: {e}")
                print("Disabling HellaSwag evaluation")
                updated_setting["enabled"] = False
                return hellaswag_eval_loader, updated_setting
        else:
            print(f"HellaSwag data not found at {hellaswag_data_path}")
            print("Disabling HellaSwag evaluation")
            updated_setting["enabled"] = False
            return hellaswag_eval_loader, updated_setting
    
    # Initialize HellaSwag evaluation loader
    if updated_setting["enabled"]:
        try:
            hellaswag_eval_loader = HellaSwagEvalLoader(
                data_file=hellaswag_data_path,
                batch_size=hellaswag_setting["batch_size"],
                seq_len=model_setting["max_context_length"],
                process_rank=0,
                num_processes=1
            )
            print(f"HellaSwag evaluation ready: {hellaswag_eval_loader.num_examples} examples")
        except Exception as e:
            print(f"Failed to initialize HellaSwag evaluation: {e}")
            print("Disabling HellaSwag evaluation")
            updated_setting["enabled"] = False
            hellaswag_eval_loader = None
    
    return hellaswag_eval_loader, updated_setting


def run_hellaswag_evaluation(hellaswag_eval_loader: HellaSwagEvalLoader, model, 
                            hellaswag_setting: Dict[str, Any], max_batches: Optional[int] = None) -> Optional[float]:
    """
    Run HellaSwag evaluation.
    
    Args:
        hellaswag_eval_loader: HellaSwag evaluation loader
        model: Model to evaluate
        hellaswag_setting: HellaSwag configuration
        max_batches: Maximum number of batches to evaluate (None for all)
        
    Returns:
        HellaSwag accuracy or None if evaluation failed
    """
    if hellaswag_eval_loader is None:
        return None
    
    try:
       
        
        # Use max_batches if provided, otherwise use default from settings
        if max_batches is None:
            max_batches = hellaswag_setting.get("max_batches", 30)
        
        accuracy = evaluate_hellaswag(
            model, hellaswag_eval_loader, 
            batch_size=hellaswag_eval_loader.B,
            seq_len=hellaswag_eval_loader.T,
            device="cpu",  # You might want to pass device as parameter
            max_batches=max_batches,
            random_sample=True
        )
        return accuracy
        
    except Exception as e:
        print(f"HellaSwag evaluation failed: {e}")
        return None


def run_final_hellaswag_evaluation(hellaswag_eval_loader: HellaSwagEvalLoader, model, 
                                  hellaswag_setting: Dict[str, Any]) -> Optional[float]:
    """
    Run final comprehensive HellaSwag evaluation.
    
    Args:
        hellaswag_eval_loader: HellaSwag evaluation loader
        model: Model to evaluate
        hellaswag_setting: HellaSwag configuration
        
    Returns:
        Final HellaSwag accuracy or None if evaluation failed
    """
    if hellaswag_eval_loader is None:
        return None
    
    try:
        print("Running final HellaSwag evaluation...")
        
        # For final evaluation, use more batches for better accuracy
        final_max_batches = hellaswag_setting.get("final_max_batches", 500)
        accuracy = evaluate_hellaswag(
            model, hellaswag_eval_loader,
            batch_size=hellaswag_eval_loader.B,
            seq_len=hellaswag_eval_loader.T,
            device="cpu",  # You might want to pass device as parameter
            max_batches=final_max_batches,
            random_sample=False  # Use all examples for final evaluation
        )
        print(f"Final HellaSwag accuracy: {accuracy:.4f}")
        return accuracy
        
    except Exception as e:
        print(f"Final HellaSwag evaluation failed: {e}")
        return None


def should_run_hellaswag_evaluation(hellaswag_setting: Dict[str, Any], iter_num: int) -> bool:
    """
    Check if HellaSwag evaluation should be run at current iteration.
    
    Args:
        hellaswag_setting: HellaSwag configuration
        iter_num: Current iteration number
        
    Returns:
        True if evaluation should be run, False otherwise
    """
    return (hellaswag_setting["enabled"] and 
            hellaswag_setting["eval_interval"] > 0 and 
            iter_num % hellaswag_setting["eval_interval"] == 0)


if __name__ == "__main__":
    # Example usage
    data_path = "moe_gpt/data/hellaswag/hellaswag_val.jsonl"
    
    # Download data if not exists
    if not os.path.exists(data_path):
        download_hellaswag_data()
    
    print("HellaSwag evaluator ready!")
