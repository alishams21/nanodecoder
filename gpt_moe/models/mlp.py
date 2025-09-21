import torch.nn as nn
from .activations import GELU


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        
