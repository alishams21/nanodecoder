
import torch
import torch.nn as nn
from torch.nn import functional as F

class Normalization(nn.Module):
    def __init__(self, e_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(e_dim))
        self.bias = nn.Parameter(torch.zeros(e_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
