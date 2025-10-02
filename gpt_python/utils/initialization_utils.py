import torch
import torch.nn as nn
import math
import sys
import os
import torch
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.experts import MLPExperts
from models.normalization import Normalization
from torch.nn import Embedding
from models.attention import MultiHeadAttention
from models.transformer import TransformerBlock

@torch.no_grad()
def init_weights(module, config, apply_gpt2_scaling=False):
    """
    Initialize weights for different module types.
    
    Args:
        module: PyTorch module to initialize
        config: Configuration object with initialization settings
        apply_gpt2_scaling: Whether to apply GPT-2 residual scaling
    """
    # optionally use switch transformer-style initialization
    # see page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
    if isinstance(module, nn.Linear):
        if config["use_switch_tfm_init"]:
            scale = config["switch_tfm_init_scale"]

            # linear layers have flipped dimensions in torch
            # size of weights is [out_dim, in_dim] 
            w_fan_in = module.weight.shape[-1]
            w_std = (scale / w_fan_in) ** 0.5
            torch.nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=w_std,
                a=-2*w_std,
                b=2*w_std,
            )
        else:
            # perform standard (normal) initialization of weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # always initialize bias to zero
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, MLPExperts):  # You'll need to import this
        # we have to init expert weights manually because
        # nn.Parameter is not a type of module in torch
        if config["use_switch_tfm_init_experts"]:
            scale = config["switch_tfm_init_scale_experts"]

            c_fc_fan_in = module.c_fc.shape[-2]
            c_fc_std = (scale / c_fc_fan_in) ** 0.5
            torch.nn.init.trunc_normal_(
                module.c_fc,
                mean=0.0,
                std=c_fc_std,
                a=-2*c_fc_std,
                b=2*c_fc_std,
            )

            c_proj_fan_in = module.c_proj.shape[-2]
            c_proj_std = (scale / c_proj_fan_in) ** 0.5
            torch.nn.init.trunc_normal_(
                module.c_proj,
                mean=0.0,
                std=c_proj_std,
                a=-2*c_proj_std,
                b=2*c_proj_std,
            )
        else:
            # perform standard (normal) initialization of weights
            torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

        # bias is always initialized to zero
        if module.fc_bias is not None:
            torch.nn.init.zeros_(module.fc_bias)
            torch.nn.init.zeros_(module.proj_bias)
    elif isinstance(module, Embedding):
        # just use standard initialization scheme for embedding always
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, Normalization):
        # Layer normalization: weight=1.0, bias=0.0
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    # Apply GPT-2 scaling if requested
    if apply_gpt2_scaling:
        if hasattr(module, 'weight') and module.weight is not None:
            param_name = None  # You'd need to track this somehow
            if param_name and (param_name.endswith('c_proj.weight') or 
                             param_name.endswith('experts.c_proj') or 
                             param_name.endswith('out_proj.weight')):
                torch.nn.init.normal_(module.weight, mean=0.0, 
                                    std=0.02/math.sqrt(2 * config["n_blocks"]))

def apply_gpt2_residual_scaling(model, config):
    """
    Apply GPT-2 special scaling to residual projection layers.
    This should be called AFTER the general initialization.
    """
    for pn, p in model.named_parameters():
        if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj') or pn.endswith('out_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config["n_blocks"]))

def initialize_model(model, config):
    """
    Complete model initialization including GPT-2 scaling.
    """
    # Apply general initialization
    model.apply(lambda module: init_weights(module, config))
    
    # Apply GPT-2 special scaling to residual projections
    for pn, p in model.named_parameters():
        if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj') or pn.endswith('out_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config["n_blocks"]))