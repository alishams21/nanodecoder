
import torch.nn as nn
from .attention import MultiHeadAttention
from .moe_layer import MOELayer
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
            qkv_bias=config["qkv_bias"],
            flash_self_attention=config["flash_self_attention"])
        self.moe_layer = MOELayer(config)
        self.att_norm = Normalization(config)
        self.moe_layer_norm = Normalization(config)
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
        x = self.moe_layer_norm(x)  # (B,T,C)
        x = self.moe_layer(x)  # (B,T,C)
        x = x + residual_connection  # (B,T,C)

        return x 
