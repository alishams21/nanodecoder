import os
import pickle
from typing import Optional, Dict, Any


def get_vocab_size_from_meta(data_dir: str) -> Optional[int]:
    """
    Get vocabulary size from meta.pkl file if it exists.
    
    Args:
        data_dir: Directory containing the meta.pkl file
        
    Returns:
        Vocabulary size if meta.pkl exists and contains vocab_size, None otherwise
    """
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta.get('vocab_size')
    return None


def load_and_update_vocab_size(data_dir: str, model_setting: Dict[str, Any]) -> int:
    """
    Load vocabulary size from meta.pkl and update model settings.
    
    Args:
        data_dir: Directory containing the meta.pkl file
        model_setting: Model configuration dictionary to update
        
    Returns:
        Final vocabulary size to use
    """
    meta_vocab_size = get_vocab_size_from_meta(data_dir)
    
    if meta_vocab_size is not None:
        print(f"Found vocab_size = {meta_vocab_size}")
        model_setting["vocab_size"] = meta_vocab_size
        return meta_vocab_size
    else:
        return model_setting["vocab_size"]


def get_vocab_size(data_dir: str, model_setting: Dict[str, Any]) -> int:
    """
    Get the vocabulary size, preferring meta.pkl over model settings.
    
    Args:
        data_dir: Directory containing the meta.pkl file
        model_setting: Model configuration dictionary
        
    Returns:
        Vocabulary size to use
    """
    meta_vocab_size = get_vocab_size_from_meta(data_dir)
    return meta_vocab_size if meta_vocab_size is not None else model_setting["vocab_size"]
