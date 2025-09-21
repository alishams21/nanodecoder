import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.initialization_utils import init_weights, apply_gpt2_residual_scaling
from utils.params_util import print_model_info
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
        assert config["n_head"] > 0, "n_head must be greater than 0"
        assert config["bias"] == True or config["bias"] == False, "bias must be a boolean"
        

        if config["n_exp"] == 1:
            blocks = nn.ModuleList(
                *[TransformerBlock(config) for _ in range(config["n_blocks"])]) # (B,T,C)
        else:
            blocks = []
            for i in range(config["n_blocks"]):
                use_moe = (i % config["stride"]) == 0
                blocks.append(TransformerBlock(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)
            
        self.trf_blocks = nn.ModuleDict({
            "tok_emb": nn.Embedding(config["vocab_size"], config["n_embd"]),
            "pos_emb": nn.Embedding(config["max_context_length"], config["n_embd"]),
            "drop_emb": nn.Dropout(config["drop_rate"]),
            "transformer_blocks": blocks,
            "final_norm": Normalization(config),
        })
        self.out_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False) # (C,V)
        self.trf_blocks.tok_emb.weight = self.out_head.weight # https://paperswithcode.com/method/weight-tying
        self.apply(lambda module: init_weights(module, config))
        apply_gpt2_residual_scaling(self, config)
        print_model_info(self, non_embedding=True)

    def forward(self, idx, targets=None):
        b, seq_len = idx.size() # (B,T)
        device = idx.device
        tok_embeds = self.trf_blocks.tok_emb(idx) # (B,T,C)
        pos_embeds = self.trf_blocks.pos_emb(torch.arange(seq_len, device=device)) # (T,C)
        x = self.trf_blocks.drop_emb(tok_embeds + pos_embeds) # (B,T,C)
        for block in self.trf_blocks.transformer_blocks:
            x = block(x)
        x = self.trf_blocks.final_norm(x) # (B,T,C)     
        logits = self.out_head(x) # (C,V)
        return logits
