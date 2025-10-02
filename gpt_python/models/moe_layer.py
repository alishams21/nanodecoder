from .router import Router
from .experts import MLPExperts
from torch import nn
import torch

class MOELayer(nn.Module):
    def __init__(
        self,config
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()
        self.router = Router(  # (noisy) top k router
            config
        )
        self.experts = MLPExperts(  # group of MLPs (experts)
            config
        )

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() # track original shape of input
        num_tokens = (B * T)

        # pass each token through the router
        used_capacity, exp_weight, exp_mask = self.router(x)

        # flatten out the input
        x = x.view(num_tokens, n_embd)

        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        # compute expert output
        exp_out = self.experts(exp_batches) # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # similar equations are used for other MoE papers
        exp_weight = exp_weight.view(num_tokens, -1) # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd) # [n_exp * exp_capacity, n_embd] 
        output = exp_weight @ exp_out # [B * T, n_embd]
        
        # resize output before return
        return output.view(B, T, n_embd)