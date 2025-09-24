#!/usr/bin/env python3
"""
HellaSwag Data Download and Preparation Script

This script downloads and prepares HellaSwag data for evaluation.
HellaSwag is a commonsense reasoning dataset for evaluating language models.
"""

import os
import json
import urllib.request
import argparse
from typing import List, Dict


def download_hellaswag_data(data_dir: str = "moe_gpt/data/hellaswag") -> str:
    """
    Download HellaSwag validation data
    
    Args:
        data_dir: Directory to save the data
        
    Returns:
        Path to the downloaded JSONL file
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # HellaSwag validation data URL
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    jsonl_path = os.path.join(data_dir, "hellaswag_val.jsonl")
    
    if os.path.exists(jsonl_path):
        print(f"HellaSwag data already exists at {jsonl_path}")
        return jsonl_path
    
    print(f"Downloading HellaSwag data from {url}")
    print(f"Saving to {jsonl_path}")
    
    try:
        urllib.request.urlretrieve(url, jsonl_path)
        print(f"Successfully downloaded HellaSwag data to {jsonl_path}")
    except Exception as e:
        print(f"Error downloading HellaSwag data: {e}")
        raise
    
    return jsonl_path


def prepare_hellaswag_data(jsonl_path: str, output_path: str) -> int:
    """
    Convert HellaSwag JSONL data to the format expected by the evaluator
    
    Args:
        jsonl_path: Path to the original HellaSwag JSONL file
        output_path: Path to save the processed data
        
    Returns:
        Number of examples processed
    """
    examples = []
    
    print(f"Processing HellaSwag data from {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract the required fields
                example = {
                    "ctx": data["ctx"],
                    "label": data["label"],
                    "endings": data["endings"]
                }
                examples.append(example)
                
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} examples...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key {e} in line {line_num}")
                continue
    
    # Save processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Prepared HellaSwag data: {len(examples)} examples")
    print(f"Saved to {output_path}")
    
    return len(examples)


def validate_hellaswag_data(jsonl_path: str) -> bool:
    """
    Validate the HellaSwag data format
    
    Args:
        jsonl_path: Path to the JSONL file to validate
        
    Returns:
        True if valid, False otherwise
    """
    print(f"Validating HellaSwag data at {jsonl_path}")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                
                # Check required fields
                required_fields = ["ctx", "label", "endings"]
                for field in required_fields:
                    if field not in data:
                        print(f"Missing field '{field}' in line {line_num}")
                        return False
                
                # Check label is valid
                if not isinstance(data["label"], int) or data["label"] < 0 or data["label"] >= 4:
                    print(f"Invalid label {data['label']} in line {line_num}")
                    return False
                
                # Check endings is a list of 4 strings
                if not isinstance(data["endings"], list) or len(data["endings"]) != 4:
                    print(f"Invalid endings in line {line_num}: expected list of 4 strings")
                    return False
                
                for i, ending in enumerate(data["endings"]):
                    if not isinstance(ending, str):
                        print(f"Invalid ending {i} in line {line_num}: expected string")
                        return False
                
                if line_num % 1000 == 0:
                    print(f"Validated {line_num} examples...")
        
        print(f"Validation successful: {line_num} examples")
        return True
        
    except Exception as e:
        print(f"Error validating data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and prepare HellaSwag data")
    parser.add_argument("--data_dir", type=str, default="moe_gpt/data/hellaswag",
                       help="Directory to save the data")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate existing data, don't download")
    parser.add_argument("--output_file", type=str, default="hellaswag_val.jsonl",
                       help="Output filename for processed data")
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    jsonl_path = os.path.join(data_dir, "hellaswag_val.jsonl")
    output_path = os.path.join(data_dir, args.output_file)
    
    if args.validate_only:
        if not os.path.exists(jsonl_path):
            print(f"Data file not found: {jsonl_path}")
            return 1
        
        if validate_hellaswag_data(jsonl_path):
            print("Data validation successful!")
            return 0
        else:
            print("Data validation failed!")
            return 1
    
    # Download data if not exists
    if not os.path.exists(jsonl_path):
        try:
            download_hellaswag_data(data_dir)
        except Exception as e:
            print(f"Failed to download data: {e}")
            return 1
    
    # Validate downloaded data
    if not validate_hellaswag_data(jsonl_path):
        print("Downloaded data validation failed!")
        return 1
    
    # Prepare data for evaluation
    try:
        num_examples = prepare_hellaswag_data(jsonl_path, output_path)
        print(f"Successfully prepared {num_examples} HellaSwag examples")
        return 0
    except Exception as e:
        print(f"Failed to prepare data: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
