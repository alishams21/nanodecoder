import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import yaml
from pathlib import Path
from src.utils.instruction_head_utils import custom_collate_fn, InstructionDataset
from src.models.gpt import GPT
from src.utils.generation import generate
from src.utils.tokenizer import text_to_token_ids, token_ids_to_text
import torch
import tiktoken
import json
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from src.utils.instruction_head_utils import format_input


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

def load_instruction_data(file_path):
    """Load instruction data from local JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instruction data file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data


# Load configuration
config = load_config()

# Get data configuration
data_config = config["data"]
file_path = data_config["file_path"]
data = load_instruction_data(file_path)
print("Number of entries:", len(data))

# Get split ratios from config
train_ratio = data_config["train_ratio"]
train_portion = int(len(data) * train_ratio)
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model selection from config
model_selection = config["model_selection"]
CHOOSE_MODEL = model_selection["choose_model"]

# Get model configuration
BASE_CONFIG = get_model_config(config, CHOOSE_MODEL)

# Get training configuration
training_config = config["training"]
num_workers = training_config["num_workers"]
batch_size = training_config["batch_size"]

# Set up collate function with config max length
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=BASE_CONFIG["max_context_length"]
)

torch.manual_seed(training_config["seed"])

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

model = GPT(BASE_CONFIG)
# Get output configuration
output_config = config["output"]
checkpoint = torch.load(output_config["model_save_path"], map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
    
print("Generating responses")
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["max_context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

# Use a default output name or add it to config if needed
test_data_path = "instruction_head_evaluation_results.json"
with open(test_data_path, "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
print(f"Responses saved as {test_data_path}")

