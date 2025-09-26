import torch
import torch.nn as nn
import sys
import os
import inspect

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.initialization_utils import init_weights, apply_gpt2_residual_scaling
from utils.params_util import print_model_info, get_num_params
from .transformer import TransformerBlock
from .normalization import Normalization
from torch.nn import functional as F
from utils.manager import MANAGER

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
        
        self.config = config
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
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.out_head(x) # (C,V)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.config["n_exp"] > 1 and self.config["use_aux_loss"]:
                loss += self.config["aux_loss_weight"] * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()
            if self.config["n_exp"] > 1 and self.config["use_router_z_loss"]:
                loss += self.config["router_z_loss_weight"] * MANAGER.aggregate_router_z_loss()
                MANAGER.reset_router_z_loss()
        else:
            logits = self.out_head(x) # (C,V)
            loss = None

        return logits, loss

    def model_surgery(self, block_size):
        assert block_size <= self.config["max_context_length"]
        self.config["max_context_length"] = block_size
        self.trf_blocks.pos_emb.weight = nn.Parameter(self.trf_blocks.pos_emb.weight[:block_size])
        for block in self.trf_blocks.transformer_blocks:
            if hasattr(block.multi_head_att, 'bias'):
                block.multi_head_att.bias = block.multi_head_att.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, config, device_type):
        # TODO: add expert config
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # add an extra check for "bias" string to account for bias terms in MoE layers
        decay_params = [p for n, p in param_dict.items() if (p.dim() >= 2 and not n.endswith('bias'))]
        nodecay_params = [p for n, p in param_dict.items() if (p.dim() < 2 or n.endswith('bias'))]
        optim_groups = [
            {'params': decay_params, 'weight_decay': config["weight_decay"]},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=config["learning_rate"], betas=config["betas"], **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = get_num_params(self)
        cfg = self.config
        L, H, Q, T = cfg["n_blocks"], cfg["n_head"], cfg["n_embd"]//cfg["n_head"], cfg["max_context_length"]
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu