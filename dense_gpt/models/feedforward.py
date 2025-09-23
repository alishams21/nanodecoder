import torch.nn as nn
from .activations import GELU


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
        )

    def forward(self, x):
        return self.layers(x)
