import torch
from torch import nn
from .activations import GELU

class MLPExperts(nn.Module):

    def __init__(
        self,
        config
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()
        self.bias = config["bias"]
        self.c_fc = nn.Parameter(torch.empty(config["n_exp"], config["n_embd"], 4 * config["n_embd"]))
        self.c_proj = nn.Parameter(torch.empty(config["n_exp"], 4 * config["n_embd"], config["n_embd"]))
        self.fc_bias = nn.Parameter(torch.empty(config["n_exp"], 1, 4 * config["n_embd"])) if self.config["bias"] else None
        self.proj_bias = nn.Parameter(torch.empty(config["n_exp"], 1, config["n_embd"])) if self.config["bias"] else None
        self.gelu = GELU()
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x