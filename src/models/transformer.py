
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .normalization import Normalization


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head_att = MultiHeadAttention(
            input_dim=config["n_embd"],
            output_dim=config["n_embd"],
            max_context_length=config["max_context_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"])
        self.feed_forward = FeedForward(config)
        self.att_norm = Normalization(config["n_embd"])
        self.feed_forward_norm = Normalization(config["n_embd"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        residual_connection = x  # (B,T,C)
        x = self.att_norm(x)  # (B,T,C)
        x = self.multi_head_att(x)   # (B,T,C)
        x = self.drop_shortcut(x)  # (B,T,C)
        x = x + residual_connection  # (B,T,C)

        # Shortcut connection for feed-forward block
        residual_connection = x  # (B,T,C)
        x = self.feed_forward_norm(x)  # (B,T,C)
        x = self.feed_forward(x)  # (B,T,C)
        x = self.drop_shortcut(x)  # (B,T,C)
        x = x + residual_connection  # (B,T,C)

        return x 
