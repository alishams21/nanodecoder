import torch
import tiktoken
import sys
import os
import yaml
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.tokenizer import text_to_token_ids, token_ids_to_text
from utils.generation import generate
from models.gpt import GPT

finetuned_model_path = Path("gpt_model.pth")


def load_config(config_path="src/train/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config()
MODEL_SETTING = config["model"]
OUTPUT_SETTINGS = config["output"]



model = GPT(MODEL_SETTING)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(OUTPUT_SETTINGS["model_save_path"], map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Oh my lord is"

token_ids = generate(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=25,
    context_size=MODEL_SETTING["max_context_length"],
    top_k=50,
    temperature=5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
