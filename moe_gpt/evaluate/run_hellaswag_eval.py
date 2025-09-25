#!/usr/bin/env python3
"""
Standalone HellaSwag Evaluation Script

This script can be used to run HellaSwag evaluation on a trained model
without running the full training loop.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gpt import GPT
from utils.training_utils import load_config
from utils.hellaswag_utils import HellaSwagEvalLoader, evaluate_hellaswag, download_hellaswag_data


def main():
    parser = argparse.ArgumentParser(description="Run HellaSwag evaluation on a trained model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, default="moe_gpt/config.yaml",
                       help="Path to the config file")
    parser.add_argument("--data_path", type=str, default="moe_gpt/data/hellaswag/hellaswag_val.jsonl",
                       help="Path to HellaSwag data")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--max_examples", type=int, default=100,
                       help="Maximum number of examples to evaluate (0 for all)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    model_config = config["model"]
    device_config = config["device"]
    
    # Override device if specified
    if args.device != "cpu":
        device_config["device_type"] = args.device
    
    device = torch.device(device_config["device_type"])
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create model
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check if HellaSwag data exists
    if not os.path.exists(args.data_path):
        print(f"HellaSwag data not found at {args.data_path}")
        print("Downloading HellaSwag data...")
        try:
            download_hellaswag_data(os.path.dirname(args.data_path))
            print(f"HellaSwag data downloaded to {args.data_path}")
        except Exception as e:
            print(f"Failed to download HellaSwag data: {e}")
            return 1
    
    # Create evaluation loader
    print("Setting up HellaSwag evaluation...")
    try:
        eval_loader = HellaSwagEvalLoader(
            data_file=args.data_path,
            batch_size=args.batch_size,
            seq_len=model_config["max_context_length"],
            process_rank=0,
            num_processes=1
        )
        print(f"HellaSwag evaluation ready: {eval_loader.num_examples} examples")
    except Exception as e:
        print(f"Failed to initialize HellaSwag evaluation: {e}")
        return 1
    
    # Run evaluation
    
    try:
        accuracy = evaluate_hellaswag(
            model, eval_loader, 
            args.batch_size, 
            model_config["max_context_length"], 
            device_config["device_type"]
        )
        print(f"\n{'='*50}")
        print(f"HellaSwag Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*50}")
        
        # Compare with human performance
        human_performance = 0.95
        if accuracy >= human_performance:
            print(f"🎉 Model performance ({accuracy*100:.2f}%) exceeds human performance ({human_performance*100:.2f}%)!")
        elif accuracy >= 0.8:
            print(f"✅ Good model performance ({accuracy*100:.2f}%) - approaching human level")
        elif accuracy >= 0.6:
            print(f"⚠️  Moderate model performance ({accuracy*100:.2f}%) - room for improvement")
        else:
            print(f"❌ Low model performance ({accuracy*100:.2f}%) - significant improvement needed")
        
        return 0
        
    except Exception as e:
        print(f"HellaSwag evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
