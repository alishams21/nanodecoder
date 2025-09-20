import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, max_context_length, dropout, num_heads, qkv_bias=False, flash_self_attention=False):
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.out_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads  
        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dim, output_dim)  
        self.att_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash_self_attention = flash_self_attention
        if not self.flash_self_attention:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer('mask', torch.triu(torch.ones(max_context_length, max_context_length), diagonal=1))
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(max_context_length, max_context_length))
                                        .view(1, 1, max_context_length, max_context_length))
    def forward(self, x):
        B, num_tokens, input_dim = x.shape

        keys = self.W_key(x)  # (B, num_tokens, output_dim)
        queries = self.W_query(x) # (B, num_tokens, output_dim)
        values = self.W_value(x) # (B, num_tokens, output_dim)

        keys = keys.view(B, num_tokens, self.num_heads, self.head_dim) # (B, num_tokens, num_heads, head_dim)
        values = values.view(B, num_tokens, self.num_heads, self.head_dim) # (B, num_tokens, num_heads, head_dim)
        queries = queries.view(B, num_tokens, self.num_heads, self.head_dim) # (B, num_tokens, num_heads, head_dim)

        keys = keys.transpose(1, 2) # (B, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2) # (B, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2) # (B, num_heads, num_tokens, head_dim)

        if self.flash_self_attention:
            attn_weights = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, dropout_p=0.1 if self.training else 0, is_causal=True)
        else:
            attn_scores = queries @ keys.transpose(2, 3)  # (B, num_heads, num_tokens, num_tokens)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # (num_tokens, num_tokens)
            attn_scores.masked_fill_(mask_bool, -torch.inf) # (B, num_heads, num_tokens, num_tokens)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # (B, num_heads, num_tokens, num_tokens)
            attn_weights = self.att_dropout(attn_weights) # (B, num_heads, num_tokens, num_tokens)
            y = attn_weights @ values # (B, num_heads, num_tokens, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, num_tokens, self.out_dim) # concatenate heads (B, num_tokens, output_dim)
        y = self.resid_dropout(y) # (B, num_tokens, output_dim)
        return y

            

