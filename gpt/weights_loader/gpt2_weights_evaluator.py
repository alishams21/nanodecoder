# Import individual functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.gpt2_weights_loader import download_and_load_gpt2
from src.models.gpt import GPT
import torch
from src.utils import text_to_token_ids, token_ids_to_text, generate
import tiktoken
from src.utils.gpt2_weights_loader import load_weights_into_gpt
import yaml

def load_config(config_path="src/weights_loader/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_model_config(config, model_key="gpt2_small_124m"):
    """Get specific model configuration from config."""
    model_config = config["gpt2_models"][model_key].copy()
    
    # Convert to the format expected by GPT model
    gpt_config = {
        "vocab_size": model_config["vocab_size"],
        "max_context_length": model_config["context_length"],
        "n_embd": model_config["n_embd"],
        "n_heads": model_config["n_heads"],
        "n_blocks": model_config["n_blocks"],
        "drop_rate": model_config["drop_rate"],
        "qkv_bias": model_config["qkv_bias"]
    }
    
    return gpt_config

config = load_config()
model_key = "gpt2_small_124m"  
MODEL_CONFIG = get_model_config(config, model_key)
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt = GPT(MODEL_CONFIG)
gpt.eval()

settings, params = download_and_load_gpt2("124M", "models")
gpt2 = load_weights_into_gpt(gpt, params)
gpt2.to(device)

torch.manual_seed(123)

token_ids = generate(
    model=gpt2,
    idx=text_to_token_ids("On the way to the success we should", tokenizer).to(device),
    max_new_tokens=25,
    context_size=MODEL_CONFIG["max_context_length"],
    top_k=50,
    temperature=5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
