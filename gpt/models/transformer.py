
import torch.nn as nn
from .attention import MultiHeadAttention
from .moe_layer import MOELayer
from .normalization import Normalization
from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, config, use_moe=False):
        super().__init__()
        self.multi_head_att = MultiHeadAttention(config)
        if use_moe:
            self.mlp = MOELayer(config)
        else:
            self.mlp = MLP(config)
        self.att_norm = Normalization(config)
        self.feed_forward_norm = Normalization(config)

    def forward(self, x):
        # Shortcut connection for attention block
        x = x + self.multi_head_att(self.att_norm(x))

        # Shortcut connection for feed-forward block
        x = x + self.mlp(self.feed_forward_norm(x))

        return x 
