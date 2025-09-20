import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .normalization import Normalization


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["vocab_size"] > 0, "vocab_size must be greater than 0"
        assert config["n_embd"] > 0, "n_embd must be greater than 0"
        assert config["max_context_length"] > 0, "max_context_length must be greater than 0"
        assert config["drop_rate"] >= 0 and config["drop_rate"] <= 1, "drop_rate must be between 0 and 1"
        assert config["n_blocks"] > 0, "n_blocks must be greater than 0"
        assert config["n_heads"] > 0, "n_heads must be greater than 0"
        assert config["bias"] == True or config["bias"] == False, "bias must be a boolean"
        
        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"]) # (V,C)
        self.pos_emb = nn.Embedding(config["max_context_length"], config["n_embd"]) # (T,C)
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_blocks"])]) # (B,T,C)

        self.final_norm = Normalization(config) # (B,T,C)
        self.out_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False) # (C,V)

    def forward(self, in_idx):
        b, seq_len = in_idx.shape # (B,T)
        tok_embeds = self.tok_emb(in_idx) # (B,T,C)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # (T,C)
        x = tok_embeds + pos_embeds  # (B,T,C)
        x = self.drop_emb(x) # (B,T,C)
        x = self.trf_blocks(x) # (B,T,C)
        x = self.final_norm(x) # (B,T,C)
        logits = self.out_head(x) # (C,V)
        return logits
