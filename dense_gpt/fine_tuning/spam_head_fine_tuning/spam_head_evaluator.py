
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from pathlib import Path
from src.utils.classification_head_utils import classify_review
from src.models.gpt import GPT
import torch
import tiktoken
import yaml

finetuned_model_path = Path("fine_tuned_with_spam_head.pth")


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_model_config(config, model_key):
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

# Initialize tokenizer and device
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get model selection from config
model_selection = config["model_selection"]
CHOOSE_MODEL = model_selection["choose_model"]
INPUT_PROMPT = model_selection["input_prompt"]

# Get model configuration
BASE_CONFIG = get_model_config(config, CHOOSE_MODEL)



model = GPT(BASE_CONFIG)

# Convert model to classifier as in section 6.5 in ch06.ipynb
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["n_embd"], out_features=num_classes)

# Then load pretrained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("fine_tuned_with_spam_head.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
text_1 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

result = classify_review(
    text_1, model, tokenizer, device, max_length=120
)

print(result)
